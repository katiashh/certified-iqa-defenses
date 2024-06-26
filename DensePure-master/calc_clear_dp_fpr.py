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
from time import time
#os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)

from pathlib import Path
import os
import sys
sys.path.append(os.path.join(Path(__file__).parent, "../ccrf/src"))


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


import os
import torch
import torch.nn as nn
from torchvision import transforms


from src.model import IQANet

class MetricModel(torch.nn.Module):
    def __init__(self, device, model_path):
        super().__init__()
        self.device = device
        model = IQANet(weighted=True).to(device)

        model.load_state_dict(torch.load(model_path, map_location=device)['state_dict'])
        model.eval().to(device)

        self.model = model
        self.lower_better = False

    def forward(self, image, inference=False):


        patch_size = 64
        patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size).reshape(image.shape[0], -1, 3, patch_size, patch_size)
        patches = patches.to(self.device)
        torch.backends.cudnn.enabled = False
        out = self.model(
            patches, patches
        ).mean()
        torch.backends.cudnn.enabled = True
        if inference:
            return out.detach().cpu().numpy()[0][0].item()
        else:
            return out



class MetricClassifier(nn.Module):
    def __init__(self):
      super().__init__()
      self.model = MetricModel(device, '../model_best_kadid_0.pkl')
      self.diap = 77.04672241210938 - 47.22482681274414

    def forward(self, x):
      #tx = torch.from_numpy(x).to(device).permute(0, 3, 1, 2)
      scores = [self.model(x)]
      #print(scores)

      N = 10
      d = self.diap / N
      new_scores = []
      for s in scores:
        b = 47.22482681274414
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

from math import ceil

import numpy as np
from scipy.stats import norm
from scipy.stats import binomtest as binom_test
from statsmodels.stats.proportion import proportion_confint
import torch

class Smooth(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, num_classes: int, sigma: float):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma

    def certify(self, x: torch.tensor, n0: int, n: int, sample_id:int, alpha: float, batch_size: int, clustering_method='none') -> (int, float):
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
        counts_selection, n0_predictions = self._sample_noise(x, n0, batch_size, clustering_method, sample_id)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation, n_predictions = self._sample_noise(x, n, batch_size, clustering_method, sample_id)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        if pABar < 0.5:
            return Smooth.ABSTAIN, 0.0, n0_predictions, n_predictions
        else:
            radius = self.sigma * norm.ppf(pABar)
            return cAHat, radius, n0_predictions, n_predictions

    def certify_noapproximate(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int) -> (int, float):
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
        top2 = counts_estimation.argsort()[::-1][:2]
        nA = counts_estimation[top2[0]].item()
        nB = counts_estimation[top2[1]].item()

        pABar = self._lower_confidence_bound(nA, n, alpha)
        pBBar = self._upper_confidence_bound(nB, n, alpha)
        if pABar < 0.5:
            return Smooth.ABSTAIN, 0.0
        else:
            radius = self.sigma/2 * (norm.ppf(pABar) - norm.ppf(pBBar))
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

    def _sample_noise(self, x: torch.tensor, num: int, batch_size, clustering_method='none', sample_id=None) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        with torch.no_grad():
            predictions_all = np.array([], dtype=int)
            counts = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1, 1, 1))
                noise = torch.randn_like(batch, device='cuda') * self.sigma

                if clustering_method == 'classifier':
                    predictions = self.base_classifier(batch + noise, sample_id)#.argmax(1)
                    print(predictions)
                    predictions = predictions.view(this_batch_size,-1).cpu().numpy()
                    count_max_list = np.zeros(this_batch_size,dtype=int)
                    for i in range(this_batch_size):
                        count_max = max(list(predictions[i]),key=list(predictions[i]).count)
                        count_max_list[i] = count_max
                    counts += self._count_arr(count_max_list, self.num_classes)

                else:
                    predictions = self.base_classifier(batch + noise, sample_id)#.argmax(1)
                    #print(predictions)
                    counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
                    predictions_all = np.hstack((predictions_all, predictions.cpu().numpy()))

            return counts, predictions_all

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

    def _upper_confidence_bound(self, NA: int, N: int, alpha: float) -> float:

        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[1]

ssigma = 0.12
num_t_steps = 2
t_total = 100
reverse_seed = 0


class SubConfig:
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
    timestep_respacing='1000'
    use_kl=False
    predict_xstart=False
    rescale_timesteps=True
    rescale_learned_sigmas=False

import os
import random

import torch
import torchvision.utils as tvu

from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
import math
import numpy as np


import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from runners.diffpure_ddpm_densepure import Diffusion
from runners.diffpure_guided_densepure import GuidedDiffusion
import torchvision.utils as tvu
import timm
from networks import *

class GuidedDiffusion(torch.nn.Module):
    def __init__(self, device=None, model_dir='pretrained'):
        super().__init__()
        #self.config = config
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device
        self.reverse_state = None
        self.reverse_state_cuda = None

        # load model
        model_config = SubConfig()
        #model_config.update(vars(self.config.model))
        print(f'model_config: {model_config}')
        model, diffusion = create_model_and_diffusion( **args_to_dict(SubConfig(), model_and_diffusion_defaults().keys()))
        model.load_state_dict(torch.load(f'../256x256_diffusion_uncond.pt'))
        #print('****')
        model.requires_grad_(False).eval().to(self.device)

        #if model_config['use_fp16']:
        #    model.convert_to_fp16()

        self.model = model
        self.model.eval()
        self.diffusion = diffusion
        self.betas = diffusion.betas
        alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        sigma = ssigma

        a = 1/(1+(sigma*2)**2)
        self.scale = a**0.5

        sigma = sigma*2
        T = t_total
        for t in range(len(self.sqrt_recipm1_alphas_cumprod)-1):
            if self.sqrt_recipm1_alphas_cumprod[t]<sigma and self.sqrt_recipm1_alphas_cumprod[t+1]>=sigma:
                if sigma - self.sqrt_recipm1_alphas_cumprod[t] > self.sqrt_recipm1_alphas_cumprod[t+1] - sigma:
                    self.t = t+1
                    break
                else:
                    self.t = t
                    break
            self.t = len(diffusion.alphas_cumprod)-1

    def image_editing_sample(self, img=None, bs_id=0, tag=None, sigma=0.0):
        assert isinstance(img, torch.Tensor)
        batch_size = img.shape[0]

        with torch.no_grad():
            if tag is None:
                tag = 'rnd' + str(random.randint(0, 10000))

            assert img.ndim == 4, img.ndim
            x0 = img

            x0 = self.scale*(img)
            t = self.t

            self.model.eval()

            if True:
                #save random state
                if True:
                    global_seed_state = torch.random.get_rng_state()
                    if torch.cuda.is_available():
                        global_cuda_state = torch.cuda.random.get_rng_state_all()

                    if self.reverse_state==None:
                        torch.manual_seed(reverse_seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(reverse_seed)
                    else:
                        torch.random.set_rng_state(self.reverse_state)
                        if torch.cuda.is_available():
                            torch.cuda.random.set_rng_state_all(self.reverse_state_cuda)

                # t steps denoise
                inter = t/num_t_steps
                indices_t_steps = [round(t-i*inter) for i in range(num_t_steps)]

                for i in range(len(indices_t_steps)):
                    t = torch.tensor([len(indices_t_steps)-i-1] * x0.shape[0], device=self.device)
                    real_t = torch.tensor([indices_t_steps[i]] * x0.shape[0], device=self.device)
                    with torch.no_grad():
                        out = self.diffusion.p_sample(
                            self.model,
                            x0,
                            t,
                            clip_denoised=True,
                            indices_t_steps = indices_t_steps.copy(),
                            T = t_total,
                            step = len(indices_t_steps)-i,
                            real_t = real_t
                        )
                        x0 = out["sample"]

                #load random state
                if True:
                    self.reverse_state = torch.random.get_rng_state()
                    if torch.cuda.is_available():
                        self.reverse_state_cuda = torch.cuda.random.get_rng_state_all()

                    torch.random.set_rng_state(global_seed_state)
                    if torch.cuda.is_available():
                        torch.cuda.random.set_rng_state_all(global_cuda_state)

            else:
                #save random state
                if True:
                    global_seed_state = torch.random.get_rng_state()
                    if torch.cuda.is_available():
                        global_cuda_state = torch.cuda.random.get_rng_state_all()

                    if self.reverse_state==None:
                        torch.manual_seed(reverse_seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(reverse_seed)
                    else:
                        torch.random.set_rng_state(self.reverse_state)
                        if torch.cuda.is_available():
                            torch.cuda.random.set_rng_state_all(self.reverse_state_cuda)

                # full steps denoise
                indices = list(range(round(t)))[::-1]
                for i in indices:
                    t = torch.tensor([i] * x0.shape[0], device=self.device)
                    with torch.no_grad():
                        out = self.diffusion.p_sample(
                            self.model,
                            x0,
                            t,
                            clip_denoised=True,
                        )
                        x0 = out["sample"]

                #load random state
                if True:
                    self.reverse_state = torch.random.get_rng_state()
                    if torch.cuda.is_available():
                        self.reverse_state_cuda = torch.cuda.random.get_rng_state_all()

                    torch.random.set_rng_state(global_seed_state)
                    if torch.cuda.is_available():
                        torch.cuda.random.set_rng_state_all(global_cuda_state)

            return x0


class DensePure_Certify(nn.Module):
    def __init__(self):
        super().__init__()

        # image classifier
        self.classifier = MetricClassifier()

        # diffusion model
        self.runner = GuidedDiffusion(device=device)


        self.register_buffer('counter', torch.zeros(1, device=device))
        self.tag = None

    def reset_counter(self):
        self.counter = torch.zeros(1, dtype=torch.int, device=device)

    def set_tag(self, tag=None):
        self.tag = tag

    def forward(self, x, sample_id):
        counter = self.counter.item()
        if counter % 5 == 0:
            print(f'diffusion times: {counter}')

        start_time = time()
        x_re = self.runner.image_editing_sample((x - 0.5) * 2, bs_id=counter, tag=self.tag, sigma=0.12)
        minutes, seconds = divmod(time() - start_time, 60)


        #self.classifier.eval()
        out = self.classifier((x_re + 1) * 0.5)


        self.counter += 1

        return out


from tqdm import tqdm
import pandas as pd

df = pd.read_csv('res/dp_fpr.csv')

model = DensePure_Certify()
smoothed_classifier_diffpure = Smooth(model, 12, ssigma)


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
        prediction, radius, _, _ = smoothed_classifier_diffpure.certify(im, 100, 1000, None, 0.001, 10, "none")
        print(prediction, radius)
        dic[img] = [prediction, radius]

df = pd.DataFrame([], columns=['path', 'score', 'radius'])
df.to_csv('res/clear_dp_fpr.csv', index=False)

for img in dic:
  df2 = pd.DataFrame([[img, dic[img][0], dic[img][1]]], columns=['path', 'score', 'radius'])
  df = pd.concat([df2, df])


df.to_csv('res/clear_dp_fpr.csv', index=False)
