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

device = torch.device("cuda:5") if torch.cuda.is_available() else torch.device("cpu")


import os
import torch
import torch.nn as nn
from torchvision import transforms


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False) # verify bias false
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001, # value found in tensorflow
                                 momentum=0.1, # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_5b(nn.Module):

    def __init__(self):
        super(Mixed_5b, self).__init__()

        self.branch0 = BasicConv2d(192, 96, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(192, 48, kernel_size=1, stride=1),
            BasicConv2d(48, 64, kernel_size=5, stride=1, padding=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(192, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1),
            BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(192, 64, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block35(nn.Module):

    def __init__(self, scale=1.0):
        super(Block35, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(320, 32, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(320, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(320, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 48, kernel_size=3, stride=1, padding=1),
            BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1)
        )

        self.conv2d = nn.Conv2d(128, 320, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_6a(nn.Module):

    def __init__(self):
        super(Mixed_6a, self).__init__()

        self.branch0 = BasicConv2d(320, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv2d(320, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Block17(nn.Module):

    def __init__(self, scale=1.0):
        super(Block17, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(1088, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(1088, 128, kernel_size=1, stride=1),
            BasicConv2d(128, 160, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(160, 192, kernel_size=(7,1), stride=1, padding=(3,0))
        )

        self.conv2d = nn.Conv2d(384, 1088, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_7a(nn.Module):

    def __init__(self):
        super(Mixed_7a, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 288, kernel_size=3, stride=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 288, kernel_size=3, stride=1, padding=1),
            BasicConv2d(288, 320, kernel_size=3, stride=2)
        )

        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block8(nn.Module):

    def __init__(self, scale=1.0, noReLU=False):
        super(Block8, self).__init__()

        self.scale = scale
        self.noReLU = noReLU

        self.branch0 = BasicConv2d(2080, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(2080, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=(1,3), stride=1, padding=(0,1)),
            BasicConv2d(224, 256, kernel_size=(3,1), stride=1, padding=(1,0))
        )

        self.conv2d = nn.Conv2d(448, 2080, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


class InceptionResNetV2(nn.Module):

    def __init__(self, num_classes=1001):
        super(InceptionResNetV2, self).__init__()
        # Special attributs
        self.input_space = None
        self.input_size = (299, 299, 3)
        self.mean = None
        self.std = None
        # Modules
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.maxpool_5a = nn.MaxPool2d(3, stride=2)
        self.mixed_5b = Mixed_5b()
        self.repeat = nn.Sequential(
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17)
        )
        self.mixed_6a = Mixed_6a()
        self.repeat_1 = nn.Sequential(
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10)
        )
        self.mixed_7a = Mixed_7a()
        self.repeat_2 = nn.Sequential(
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20)
        )
        self.block8 = Block8(noReLU=True)
        self.conv2d_7b = BasicConv2d(2080, 1536, kernel_size=1, stride=1)
        self.avgpool_1a = nn.AvgPool2d(8, count_include_pad=False)
        self.last_linear = nn.Linear(1536, num_classes)

    def features(self, input):
        x = self.conv2d_1a(input)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.maxpool_5a(x)
        x = self.mixed_5b(x)
        x = self.repeat(x)
        x = self.mixed_6a(x)
        x = self.repeat_1(x)
        x = self.mixed_7a(x)
        x = self.repeat_2(x)
        x = self.block8(x)
        x = self.conv2d_7b(x)
        return x

    def logits(self, features):
        x = self.avgpool_1a(features)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

def inceptionresnetv2(weights_path, num_classes=1000, pretrained='imagenet'):
    r"""InceptionResNetV2 model architecture from the
    `"InceptionV4, Inception-ResNet..." <https://arxiv.org/abs/1602.07261>`_ paper.
    """
    if pretrained:

        # both 'imagenet'&'imagenet+background' are loaded from same parameters
        model = InceptionResNetV2(num_classes=1001)
        model.load_state_dict(torch.load(weights_path, map_location=lambda storage, loc: storage))

        if pretrained == 'imagenet':
            new_last_linear = nn.Linear(1536, 1000)
            new_last_linear.weight.data = model.last_linear.weight.data[1:]
            new_last_linear.bias.data = model.last_linear.bias.data[1:]
            model.last_linear = new_last_linear

        model.input_space = 'RGB'
        model.input_size = [3, 299, 299]
        model.input_range = [0, 1]

        model.mean = [0.5, 0.5, 0.5]
        model.std = [0.5, 0.5, 0.5]
    else:
        model = InceptionResNetV2(num_classes=num_classes)
    return model
    
class model_qa(nn.Module):
    def __init__(self, weights_path, num_classes,**kwargs):
        super(model_qa,self).__init__()
        base_model = inceptionresnetv2(weights_path, num_classes=1000, pretrained='imagenet')
        self.base= nn.Sequential(*list(base_model.children())[:-1])
        self.fc = nn.Sequential(
            nn.Linear(1536, 2048),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(2048),
            nn.Dropout(p=0.25),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=0.25),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),         
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self,x):
        x = self.base(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x   


class MetricModel(torch.nn.Module):
    def __init__(self, device, model_path, backbone_path=None):
        super().__init__()
        self.device = device

        model = model_qa(backbone_path, num_classes=1).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval().to(device)
        self.model = model
        self.lower_better = False
    
    def forward(self, image, inference=False):
        # transforms.Compose doesn't accept torch tensors
        out = self.model(
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(transforms.Resize([512, 384])(image))
        )
        if inference:
            return out.detach().cpu().numpy()[0][0].item()
        else:
            return out


class MetricClassifier(nn.Module):
    def __init__(self):
      super().__init__()
      self.model = MetricModel(device=device, model_path='../KonCept512.pth', backbone_path='../inceptionresnetv2-520b38e4.pth')
      self.diap = 66.86933898925781 - 26.40314292907715

    def forward(self, x):
      #tx = torch.from_numpy(x).to(device).permute(0, 3, 1, 2)
      scores = self.model(x)
      #print(scores)

      N = 10
      d = self.diap / N
      new_scores = []
      for s in scores:
        b = 26.40314292907715
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



paths = []
attacks = ['cadv', 'grad-est', 'korhonen-et-al', 'onepixel', 'patch-rs', 'ssah', 'uap',
'cnn-attack', 'ifgsm', 'madc', 'parsimonious', 'square-attack', 'stadv', 'zhang-et-al-dists']
presets = ['preset_0', 'preset_1', 'preset_2']
for preset in presets:
        for attack in attacks:
                path = os.path.join('../../../../data/DIONE/work/Framework_Datasets/dataset/attacked-dataset/no-defence/', preset, attack, 'koncept')
                files = sorted(os.listdir(path))[:10]
                for i in range(10):
                        paths.append(os.path.join(path, files[i]))


from tqdm import tqdm
import pandas as pd

df = pd.DataFrame([], columns=['path', 'score', 'radius'])
df.to_csv('res/crff12_koncept.csv', index=False)


for path in tqdm(paths):
  im = cv2.imread(path)
  im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype('float32') / 255.
  im = cv2.resize(im, (256, 256))
  im = torch.from_numpy(im).to(device).permute(2, 0, 1).to(device)
  #im = torch.nn.functional.interpolate(im, (256, 256), mode='bicubic', antialias=True)
  prediction, radius = smoothed_classifier.certify(im, 100, 1000, 0.001, 10)
  print(prediction, radius)
  df = pd.read_csv('res/crff12_koncept.csv')
  df2 = pd.DataFrame([[path, prediction, radius]], columns=['path', 'score', 'radius'])
  df = pd.concat([df2, df])
  df.to_csv('res/crff12_koncept.csv', index=False)
