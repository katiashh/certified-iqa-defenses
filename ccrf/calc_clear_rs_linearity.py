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

device = torch.device("cuda:6") if torch.cuda.is_available() else torch.device("cpu")

iqa_metric = pyiqa.create_metric('paq2piq', device=device)
model = iqa_metric

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




def SPSP(x, P=1, method='avg'):
    batch_size = x.size(0)
    map_size = x.size()[-2:]
    pool_features = []
    for p in range(1, P+1):
        pool_size = [int(d / p) for d in map_size]
        if method == 'maxmin':
            M = F.max_pool2d(x, pool_size)
            m = -F.max_pool2d(-x, pool_size)
            pool_features.append(torch.cat((M, m), 1).view(batch_size, -1))  # max & min pooling
        elif method == 'max':
            M = F.max_pool2d(x, pool_size)
            pool_features.append(M.view(batch_size, -1))  # max pooling
        elif method == 'min':
            m = -F.max_pool2d(-x, pool_size)
            pool_features.append(m.view(batch_size, -1))  # min pooling
        elif method == 'avg':
            a = F.avg_pool2d(x, pool_size)
            pool_features.append(a.view(batch_size, -1))  # average pooling
        else:
            m1  = F.avg_pool2d(x, pool_size)
            rm2 = torch.sqrt(F.relu(F.avg_pool2d(torch.pow(x, 2), pool_size) - torch.pow(m1, 2)))
            if method == 'std':
                pool_features.append(rm2.view(batch_size, -1))  # std pooling
            else:
                pool_features.append(torch.cat((m1, rm2), 1).view(batch_size, -1))  # statistical pooling: mean & std
    return torch.cat(pool_features, dim=1)


class IQAModel(nn.Module):
    def __init__(self, arch='resnext101_32x8d', features_weights_path=None, pool='avg', use_bn_end=False, P6=1, P7=1):
        super(IQAModel, self).__init__()
        self.pool = pool
        self.use_bn_end = use_bn_end
        if pool in ['max', 'min', 'avg', 'std']:
            c = 1
        else:
            c = 2
        self.P6 = P6  #
        self.P7 = P7  #
        if features_weights_path:
            backbone = models.__dict__[arch]()
            backbone.load_state_dict(torch.load(features_weights_path))
        else:
            backbone = models.__dict__[arch](pretrained=False)
        features = list(backbone.children())[:-2]
        if arch == 'alexnet':
            in_features = [256, 256]
            self.id1 = 9
            self.id2 = 12
            features = features[0]
        elif arch == 'vgg16':
            in_features = [512, 512]
            self.id1 = 23
            self.id2 = 30
            features = features[0]
        elif 'res' in arch:
            self.id1 = 6
            self.id2 = 7
            if arch == 'resnet18' or arch == 'resnet34':
                in_features = [256, 512]
            else:
                in_features = [1024, 2048]
        else:
            print('The arch is not implemented!')
        self.features = nn.Sequential(*features)
        self.dr6 = nn.Sequential(nn.Linear(in_features[0] * c * sum([p * p for p in range(1, self.P6+1)]), 1024),
                                 nn.BatchNorm1d(1024),
                                 nn.Linear(1024, 256),
                                 nn.BatchNorm1d(256),
                                 nn.Linear(256, 64),
                                 nn.BatchNorm1d(64), nn.ReLU())
        self.dr7 = nn.Sequential(nn.Linear(in_features[1] * c * sum([p * p for p in range(1, self.P7+1)]), 1024),
                                 nn.BatchNorm1d(1024),
                                 nn.Linear(1024, 256),
                                 nn.BatchNorm1d(256),
                                 nn.Linear(256, 64),
                                 nn.BatchNorm1d(64), nn.ReLU())

        if self.use_bn_end:
            self.regr6 = nn.Sequential(nn.Linear(64, 1), nn.BatchNorm1d(1))
            self.regr7 = nn.Sequential(nn.Linear(64, 1), nn.BatchNorm1d(1))
            self.regression = nn.Sequential(nn.Linear(64 * 2, 1), nn.BatchNorm1d(1))
        else:
            self.regr6 = nn.Linear(64, 1)
            self.regr7 = nn.Linear(64, 1)
            self.regression = nn.Linear(64 * 2, 1)

    def extract_features(self, x):
        f, pq = [], []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii == self.id1:
                x6 = SPSP(x, P=self.P6, method=self.pool)
                x6 = self.dr6(x6)
                f.append(x6)
                pq.append(self.regr6(x6))
            if ii == self.id2:
                x7 = SPSP(x, P=self.P7, method=self.pool)
                x7 = self.dr7(x7)
                f.append(x7)
                pq.append(self.regr7(x7))

        f = torch.cat(f, dim=1)

        return f, pq

    def forward(self, x):
        f, pq = self.extract_features(x)
        s = self.regression(f)
        pq.append(s)

        return pq


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)
    def forward(self, x):
        return (x - self.mean.type_as(x)[None,:,None,None]) / self.std.type_as(x)[None,:,None,None]




class MetricModel(torch.nn.Module):
    def __init__(self, device, model_path):
        super().__init__()
        self.device = device

        model = IQAModel(arch='resnext101_32x8d')
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        model.eval()

        self.k = checkpoint['k'][0]
        self.b = checkpoint['b'][0]
        self.model = model.to(device)
        self.lower_better = False

    def forward(self, image, inference=False):
        out = self.model(
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
        )[-1] * self.k + self.b
        if inference:
            return out.detach().cpu().numpy()[0][0].item()
        else:
            return out



class MetricClassifier(nn.Module):
    def __init__(self):
      super().__init__()
      self.model = MetricModel(device=device, model_path='../p1q2.pth')#pyiqa.create_metric('paq2piq', device=device)
      self.diap = 83.22696685791016 - 25.780681610107422

    def forward(self, x):
      #tx = torch.from_numpy(x).to(device).permute(0, 3, 1, 2)
      scores = self.model(x)
      #print(scores)

      N = 10
      d = self.diap / N
      new_scores = []
      for s in scores:
        b = 25.780681610107422
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

df = pd.read_csv('res2/rs_linearity.csv')

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
df.to_csv('res2/clear_rs_linearity.csv', index=False)

for img in dic:
  df2 = pd.DataFrame([[img, dic[img][0], dic[img][1]]], columns=['path', 'pred', 'radius'])
  df = pd.concat([df2, df])


df.to_csv('res2/clear_rs_linearity.csv', index=False)


df = pd.read_csv('res2/rs_dn_linearity.csv')

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
df.to_csv('res2/clear_rs_dn_linearity.csv', index=False)

for img in dic:
  df2 = pd.DataFrame([[img, dic[img][0], dic[img][1]]], columns=['path', 'pred', 'radius'])
  df = pd.concat([df2, df])


df.to_csv('res2/clear_rs_dn_linearity.csv', index=False)
