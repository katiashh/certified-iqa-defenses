import pyiqa
import torch
import time
import datetime
import numpy as np
from math import ceil
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint
import cv2
import os
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import timm

#os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)
from core import Smooth

class Smooth(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, num_classes: int, sigma: float, t: int):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma
        self.t = t

    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int) -> (int, float):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.
        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise(x, n0, batch_size)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(x, n, batch_size)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        if pABar < 0.5:
            return Smooth.ABSTAIN, 0.0
        else:
            radius = self.sigma * norm.ppf(pABar)
            return cAHat, radius

    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> int:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).
        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.
        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        self.base_classifier.eval()
        counts = self._sample_noise(x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]

        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return Smooth.ABSTAIN
        else:
            return top2[0]

    def _sample_noise(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.
        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1, 1, 1))

                predictions = self.base_classifier(batch, self.t)#.argmax(1)
                #print(predictions.shape)
                counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
            return counts

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.
        This function uses the Clopper-Pearson method.
        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]

class Args:
    image_size=256
    num_channels=256
    num_res_blocks=2
    num_heads=4
    num_heads_upsample=-1
    num_head_channels=64
    attention_resolutions="32,16,8"
    channel_mult=""
    dropout=0.0
    class_cond=False
    use_checkpoint=False
    use_scale_shift_norm=True
    resblock_updown=True
    use_fp16=False
    use_new_attention_order=False
    clip_denoised=True
    num_samples=10000
    batch_size=16
    use_ddim=False
    model_path=""
    classifier_path=""
    classifier_scale=1.0
    learn_sigma=True
    diffusion_steps=1000
    noise_schedule="linear"
    timestep_respacing=None
    use_kl=False
    predict_xstart=False
    rescale_timesteps=False
    rescale_learned_sigmas=False

device = torch.device("cuda:4") if torch.cuda.is_available() else torch.device("cpu")

import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import numpy as np
from PIL import Image



class Image_load(object):
    def __init__(self, size, stride, interpolation=Image.BILINEAR):
        assert isinstance(size, int)
        self.size = size
        self.stride = stride
        self.interpolation = interpolation

    def __call__(self, img):
        image = self.adaptive_resize(img)
        return self.generate_patches(image, input_size=self.stride)

    def adaptive_resize(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        h, w = img.size
        if h < self.size or w < self.size:
            return transforms.ToTensor()(img)
        else:
            return transforms.ToTensor()(transforms.Resize(self.size, self.interpolation)(img))

    def to_numpy(self, image):
        p = image.numpy()
        return p.transpose((1, 2, 0))

    def generate_patches(self, image, input_size, type=np.float32):
        img = self.to_numpy(image)
        img_shape = img.shape
        img = img.astype(dtype=type)
        if len(img_shape) == 2:
            H, W, = img_shape
            ch = 1
        else:
            H, W, ch = img_shape
        if ch == 1:
            img = np.asarray([img, ] * 3, dtype=img.dtype)

        stride = int(input_size / 2)
        hIdxMax = H - input_size
        wIdxMax = W - input_size

        hIdx = [i * stride for i in range(int(hIdxMax / stride) + 1)]
        if H - input_size != hIdx[-1]:
            hIdx.append(H - input_size)
        wIdx = [i * stride for i in range(int(wIdxMax / stride) + 1)]
        if W - input_size != wIdx[-1]:
            wIdx.append(W - input_size)
        patches_numpy = [img[hId:hId + input_size, wId:wId + input_size, :]
                         for hId in hIdx
                         for wId in wIdx]
        patches_tensor = [transforms.ToTensor()(p) for p in patches_numpy]
        patches_tensor = torch.stack(patches_tensor, 0).contiguous()
        return patches_tensor.squeeze(0)

    def generate_patches_custom(self, image, input_size, type=np.float32):
        # img = self.to_numpy(image)
        img = image
        img = img.to(torch.float32)
        img = img.squeeze(0)
        img_shape = img.shape
        ch, H, W = img_shape
        # if len(img_shape) == 2:
        #    H, W, = img_shape
        #    ch = 1
        # else:
        #    H, W, ch = img_shape
        # if ch == 1:
        #    img = np.asarray([img, ] * 3, dtype=img.dtype)

        stride = int(input_size / 2)
        hIdxMax = H - input_size
        wIdxMax = W - input_size

        hIdx = [i * stride for i in range(int(hIdxMax / stride) + 1)]
        if H - input_size != hIdx[-1]:
            hIdx.append(H - input_size)
        wIdx = [i * stride for i in range(int(wIdxMax / stride) + 1)]
        if W - input_size != wIdx[-1]:
            wIdx.append(W - input_size)
        patches_tensor = [img[:, hId:hId + input_size, wId:wId + input_size]
                          for hId in hIdx
                          for wId in wIdx]
        # patches_tensor = [transforms.ToTensor()(p) for p in patches_numpy]
        patches_tensor = torch.stack(patches_tensor, 0).contiguous()
        return patches_tensor.squeeze(0)


class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.backbone = torchvision.models.resnet50(pretrained=False)
        fc_feature = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(fc_feature, 1, bias=True)

    def forward(self, x):
        result = self.backbone(x)
        return result


class SPAQ(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

        curdir = os.path.abspath(os.path.join(__file__, os.pardir))
        model = Baseline()
        model.load_state_dict(torch.load(os.path.join(curdir, '../BL_release.pt'), map_location='cpu')['state_dict'])
        self.model = model.eval().to(device)
        self.loader = Image_load(size=512, stride=224)

    def forward(self, tensor):
        tensor = transforms.Resize(512)(tensor)
        patches = self.loader.generate_patches_custom(tensor, 224).to(self.device)
        return self.model(patches).mean()

curdir = os.path.abspath(os.path.join(__file__, os.pardir))

class MetricModel(torch.nn.Module):
    def __init__(self, device, model_path=f'../BL_release.pt'):
        super().__init__()
        self.device = device
        self.lower_better = False
        self.full_reference = False

        model = Baseline()
        model.load_state_dict(torch.load(model_path, map_location=device)['state_dict'])
        self.model = model.eval().to(device)
        self.loader = Image_load(size=512, stride=224)

    def forward(self, image, inference=False):
        patches = self.loader.generate_patches_custom(image, 224).to(self.device)
        return self.model(patches).mean()



class MetricClassifier(nn.Module):
    def __init__(self):
      super().__init__()
      self.model = MetricModel(device=device)
      self.diap = 77.75585174560547 - 21.749107360839844

    def forward(self, x):
      #tx = torch.from_numpy(x).to(device).permute(0, 3, 1, 2)
      scores = [self.model(x)]
      #print(scores.shape)

      N = 10
      d = self.diap / N
      new_scores = []
      for s in scores:
        b = 21.749107360839844
        cur = -1
        if s <= b:
                cur = 0
        for i in range(N):
          if s > b and s <= b + d:
            cur = i+1
          b += d
        if cur == -1:
          cur = N+1
        new_scores.append(cur)
      new_scores = torch.from_numpy(np.array(new_scores))
      return new_scores


class DiffusionRobustModel(nn.Module):
    def __init__(self, classifier_name="beit"):
        super().__init__()
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(Args(), model_and_diffusion_defaults().keys())
        )
        model.load_state_dict(
            torch.load("../256x256_diffusion_uncond.pt")
        )
        model.eval().to(device)

        self.model = model
        self.diffusion = diffusion

        # Load the BEiT model
        #classifier = timm.create_model('beit_large_patch16_512', pretrained=True)
        #classifier.eval().cuda()
        classifier = MetricClassifier()

        self.classifier = classifier

        #self.model = torch.nn.DataParallel(self.model).cuda()
        #self.classifier = torch.nn.DataParallel(self.classifier).cuda()

    def forward(self, x, t):
        x_in = x * 2 -1
        #x = torch.nn.functional.interpolate(x, (256, 256), mode='bicubic', antialias=True)
        imgs = self.denoise(x_in, t)
        imgs = (imgs + 1) / 2

        #imgs = torch.nn.functional.interpolate(imgs, (512, 512), mode='bicubic', antialias=True)

        imgs = torch.tensor(imgs).to(device)
        with torch.no_grad():
            out = self.classifier(imgs)

        return out

    def denoise(self, x_start, t, multistep=False):
        t_batch = torch.tensor([t] * len(x_start)).to(device)

        noise = torch.randn_like(x_start)

        x_t_start = self.diffusion.q_sample(x_start=x_start, t=t_batch, noise=noise)

        with torch.no_grad():
            if multistep:
                out = x_t_start
                for i in range(t)[::-1]:
                    print(i)
                    t_batch = torch.tensor([i] * len(x_start)).to(device)
                    out = self.diffusion.p_sample(
                        self.model,
                        out,
                        t_batch,
                        clip_denoised=True
                    )['sample']
            else:
                out = self.diffusion.p_sample(
                    self.model,
                    x_t_start,
                    t_batch,
                    clip_denoised=True
                )['pred_xstart']

        return out

model = DiffusionRobustModel()
sigma = 0.12
target_sigma = sigma * 2
real_sigma = 0
t = 0
while real_sigma < target_sigma:
  t += 1
  a = model.diffusion.sqrt_alphas_cumprod[t]
  b = model.diffusion.sqrt_one_minus_alphas_cumprod[t]
  real_sigma = b / a
  smoothed_classifier = Smooth(model, 12, sigma, t)



from tqdm import tqdm
import pandas as pd

df = pd.read_csv('res/crff12_spaq.csv')

dic = {}
for i in tqdm(range(len(df))):
    img = df.iloc[i]['path'].split('/')[-1][:-4]+'.jpg'
    print(df.iloc[i]['path'])
    print(img)
    if img not in dic:
        path = os.path.join('../../../../data/DIONE/work/Framework_Datasets/dataset/quality-sampled-datasets/koniq_sampled_MOS/1000_10_clusters', img)
        print(path)
        im = cv2.imread(path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype('float32') / 255.
        im = cv2.resize(im, (256, 256))
        im = torch.from_numpy(im).to(device).permute(2, 0, 1).to(device)
        #im = torch.nn.functional.interpolate(im, (256, 256), mode='bicubic', antialias=True)
        prediction, radius = smoothed_classifier.certify(im, 100, 1000, 0.001, 1)
        print(prediction, radius)
        dic[img] = [prediction, radius]

df = pd.DataFrame([], columns=['path', 'score', 'radius'])
df.to_csv('res/clear_spaq12.csv', index=False)

for img in dic:
  df2 = pd.DataFrame([[img, dic[img][0], dic[img][1]]], columns=['path', 'score', 'radius'])
  df = pd.concat([df2, df])


df.to_csv('res/clear_spaq12.csv', index=False)
