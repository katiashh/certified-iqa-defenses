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
import pandas as pd
import torch
import torch.nn as nn
import timm

device = torch.device("cuda:5") if torch.cuda.is_available() else torch.device("cpu")


from pathlib import Path
import os
import sys
sys.path.append(os.path.join(Path(__file__).parent, ".."))

from architectures import get_architecture, IMAGENET_CLASSIFIERS
from datasets import get_dataset, DATASETS
from torch.nn import MSELoss, CrossEntropyLoss
from torch.optim import SGD, Optimizer, Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from train_utils import AverageMeter, accuracy, init_logfile, log

import argparse
import datetime
import numpy as np
import os
import time
import torch
from archs.dncnn import DnCNN
from collections import OrderedDict

checkpoint = torch.load('../best.pth', map_location=device)
f = OrderedDict()
for k in checkpoint['state_dict']:
  f[k[7:]] = checkpoint['state_dict'][k]
denoiser = DnCNN(image_channels=3, depth=17, n_channels=64).to(device)
denoiser.load_state_dict(f)
denoiser.eval()

sigma = 0.12
alpha = 0.001

import math
import time
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pdb
import scipy.stats as stats

def to_cpu(tensor):
    return tensor.detach().cpu()

import math
import time
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pdb
import scipy.stats as stats

def to_cpu(tensor):
    return tensor.detach().cpu()
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms


import math

import torch
import torch.nn as nn
from torchvision.models import resnet18

from torchvision import transforms


class BaselineModel1(nn.Module):
    def __init__(self, num_classes, keep_probability, inputsize):

        super(BaselineModel1, self).__init__()
        self.fc1 = nn.Linear(inputsize, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop_prob = (1 - keep_probability)
        self.relu1 = nn.PReLU()
        self.drop1 = nn.Dropout(self.drop_prob)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.PReLU()
        self.drop2 = nn.Dropout(p=self.drop_prob)
        self.fc3 = nn.Linear(512, num_classes)
        self.sig = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Weight initialization reference: https://arxiv.org/abs/1502.01852
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.weight.data.normal_(0, 0.02)
            #     m.bias.data.zero_()

    def forward(self, x):
        """
        Feed-measure pass.
        :param x: Input tensor
        : return: Output tensor
        """
        out = self.fc1(x)

        out = self.bn1(out)
        out = self.relu1(out)
        out = self.drop1(out)
        out = self.fc2(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.drop2(out)
        out = self.fc3(out)
        out = self.sig(out)
        # out_a = torch.cat((out_a, out_p), 1)

        # out_a = self.sig(out)
        return out


class MetaIQA(nn.Module):
    def __init__(self, device, model_path):
        super().__init__()

        self.resnet_layer = resnet18(pretrained=False)
        self.net = BaselineModel1(1, 0.5, 1000)


    def forward(self, x):
        x = self.resnet_layer(x)
        x = self.net(x)
        return x




class MetricModel(torch.nn.Module):
    def __init__(self, device, model_path):
        super().__init__()
        self.device = device

        model = MetaIQA(device, model_path)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)

        model.eval().to(device)
        self.model = model
        self.lower_better = False

    def forward(self, image, inference=False):
        # transforms.Compose doesn't accept torch tensors
        out = self.model(
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
        )
        if inference:
            return out.detach().cpu().numpy()[0][0].item()
        else:
            return out



class MetricClassifier(nn.Module):
    def __init__(self):
      super().__init__()
      self.model = MetricModel(device, '../metaiqa.pth')
      self.diap = 1.0 - 0.0

    def forward(self, x):
      #tx = torch.from_numpy(x).to(device).permute(0, 3, 1, 2)
      scores = self.model(x)
      #print(scores)

      N = 10
      d = self.diap / N
      new_scores = []
      for s in scores:
        b = 0.0
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

clf = MetricClassifier()

def noisy_samples(x, n):
        """
        Adds Gaussian noise to `x` to generate samples. Optionally augments `y` similarly.

        :param x: Sample input with shape as expected by the model.
        :param n: Number of noisy samples to create.
        :return: Array of samples of the same shape as `x`.
        """

        # augment x
        x = np.expand_dims(x, axis=0)
        x = np.repeat(x, n, axis=0)
        x = x + np.random.normal(scale=sigma, size=x.shape).astype('float32')
        x[x > 1] = 1
        x[x < 0] = 0

        return x

def predict_classifier(x, dn=False):
  tx = torch.from_numpy(x).to(device).permute(0, 3, 1, 2)
  #print('***')
  #print(tx.shape)
  if dn:
    tx = denoiser(tx)

  scores = clf(tx)
  return scores


def prediction_counts(x, n, batch_size, dn=False):
        """
        Makes predictions and then converts probability distribution to counts.

        :param x: Sample input with shape as expected by the model.
        :param n: Number of noisy samples to create.
        :param batch_size: Size of batches.
        :return: Array of counts with length equal to number of columns of `x`.
        """
        # sample and predict
        preds = []
        clear_pred =  predict_classifier(x=x[None,:], dn=dn)[0]
        limit = 10000

        counter = 0
        for i in range(n // batch_size):
            with torch.no_grad():
                x_new = noisy_samples(x, n=batch_size)
                predictions = predict_classifier(x=x_new, dn=dn)
                for el in predictions:
                    preds.append(el)

        return np.array(preds)[:n]


def lower_confidence_bound(n_class_samples: int, n_total_samples: int) -> float:
        """
        Uses Clopper-Pearson method to return a (1-alpha) lower confidence bound on bernoulli proportion

        :param n_class_samples: Number of samples of a specific class.
        :param n_total_samples: Number of samples for certification.
        :return: Lower bound on the binomial proportion w.p. (1-alpha) over samples.
        """
        from statsmodels.stats.proportion import proportion_confint

        return proportion_confint(n_class_samples, n_total_samples, alpha=2 * alpha, method="beta")[0]


def certify_clf(x: np.ndarray, n: int, batch_size: int = 32, dn=False):
        """
        Computes certifiable radius around input `x` and returns radius `r` and prediction.

        :param x: Sample input with shape as expected by the model.
        :param n: Number of samples for estimate certifiable radius.
        :param batch_size: Batch size.
        :return: Tuple of length 2 of the selected class and certified radius.
        """
        prediction = []
        radius = []

        for x_i in x:

            # get sample prediction for classification
            scores = prediction_counts(x_i, n=100, batch_size=batch_size, dn=dn)
            if scores is None:
              return [-1], [0]
            unique, counts = np.unique(scores, return_counts=True)
            class_select = int(unique[int(np.argmax(counts))])

            # get sample prediction for certification
            scores = prediction_counts(x_i, n=n, batch_size=batch_size, dn=dn)
            if scores is None:
              return [-1], [0]
            unique, counts = np.unique(scores, return_counts=True)
            id = -1
            for i in range(len(unique)):
              if unique[i] == class_select:
                id = i
            count_class = counts[id]

            prob_class = lower_confidence_bound(count_class, n)

            if prob_class < 0.5:
                prediction.append(-1)
                radius.append(0.0)
            else:
                prediction.append(class_select)
                radius.append(sigma * norm.ppf(prob_class))

        return np.array(prediction), np.array(radius)


from tqdm import tqdm
import pandas as pd

df = pd.read_csv('res2/rs_meta-iqa.csv')

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
        #im = cv2.resize(im, (256, 256))
        #im = torch.from_numpy(im).to(device).permute(2, 0, 1).to(device)
        #im = torch.nn.functional.interpolate(im, (256, 256), mode='bicubic', antialias=True)
        pred, radius = certify_clf(x=im[None, :], n=1000, batch_size=10, dn=False)
        #detections, detections_l, detections_u = smoothed_model.predict_range(im, n=1000, batch_size=10, q_u=q_u, q_l=q_l, dn=False)
        print(pred, radius)
        dic[img] = [pred, radius]

df = pd.DataFrame([], columns=['path', 'pred', 'radius'])
df.to_csv('res2/clear_rs_meta-iqa.csv', index=False)

for img in dic:
  df2 = pd.DataFrame([[img, dic[img][0], dic[img][1]]], columns=['path', 'pred', 'radius'])
  df = pd.concat([df2, df])


df.to_csv('res2/clear_rs_meta-iqa.csv', index=False)


df = pd.read_csv('res2/rs_dn_meta-iqa.csv')

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
        #im = cv2.resize(im, (256, 256))
        #im = torch.from_numpy(im).to(device).permute(2, 0, 1).to(device)
        #im = torch.nn.functional.interpolate(im, (256, 256), mode='bicubic', antialias=True)
        pred, radius = certify_clf(x=im[None, :], n=1000, batch_size=10, dn=True)
        #detections, detections_l, detections_u = smoothed_model.predict_range(im, n=1000, batch_size=10, q_u=q_u, q_l=q_l, dn=False)
        print(pred, radius)
        dic[img] = [pred, radius]

df = pd.DataFrame([], columns=['path', 'pred', 'radius'])
df.to_csv('res2/clear_rs_dn_meta-iqa.csv', index=False)

for img in dic:
  df2 = pd.DataFrame([[img, dic[img][0], dic[img][1]]], columns=['path', 'pred', 'radius'])
  df = pd.concat([df2, df])


df.to_csv('res2/clear_rs_dn_meta-iqa.csv', index=False)


