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


model = MetricModel(device=device, model_path='../KonCept512.pth', backbone_path='../inceptionresnetv2-520b38e4.pth')

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

class DetectionsAcc:
    OBJECT_SORT=0
    CENTER_SORT=1
    CORNER_SORT=2
    SINGLE_BIN=0
    LABEL_BIN=1
    LOCATION_BIN=2
    LOCATION_LABEL_BIN=3

    def __init__(self, bin=SINGLE_BIN, sort=OBJECT_SORT, loc_bin_count=None):
        self.detections_list = []
        self.max_num_detections = 0
        #count the number of classes in each class bin
        self.bin_counts = {}
        self.detections_tensor = None
        self.id_index_map = {}

        self.sort = sort
        self.bin = bin
        self.loc_bin_count = loc_bin_count
    def track(self, detections):
        #dim of detections (# of simulations, tensor((#of detections, 7)))
        self.detections_list.extend(detections)
        for detection in detections:
            if detection is not None:

                temp_count = {}
                if self.bin == DetectionsAcc.SINGLE_BIN:
                    box_count = detection.size(0)
                    if box_count > self.max_num_detections:
                        self.max_num_detections = box_count
                elif (self.bin ==  DetectionsAcc.LABEL_BIN
                      or self.bin == DetectionsAcc.LOCATION_BIN
                      or self.bin == DetectionsAcc.LOCATION_LABEL_BIN):
                    if self.bin == DetectionsAcc.LABEL_BIN:
                        # for label binning
                        ids = detection[:, -1].tolist()
                    elif self.bin == DetectionsAcc.LOCATION_BIN:
                        # for location binning
                        midx = (detection[:, 0] + detection[:, 2]) / 2
                        midy = (detection[:, 1] + detection[:, 3]) / 2
                        xids = (midx/416*self.loc_bin_count).floor()
                        yids = (midy/416*self.loc_bin_count).floor()
                        ids = (xids+yids*10).tolist()
                    elif self.bin == DetectionsAcc.LOCATION_LABEL_BIN:
                        # for location+label binning
                        midx = (detection[:, 0] + detection[:, 2]) / 2
                        midy = (detection[:, 1] + detection[:, 3]) / 2
                        xids = (midx / 416 * self.loc_bin_count).floor()
                        yids = (midy / 416 * self.loc_bin_count).floor()
                        labels = detection[:, -1]
                        ids = (xids + yids * 10 + labels * 100).tolist()

                    for id in ids:
                        if id not in temp_count:
                            temp_count[id] = 1
                        else:
                            temp_count[id] += 1
                    for id, count in temp_count.items():
                        if id not in self.bin_counts:
                            self.bin_counts[id] = count
                        elif self.bin_counts[id] < count:
                            self.bin_counts[id] = count

    def tensorize(self):
        if self.bin == DetectionsAcc.SINGLE_BIN:
            self.detection_len = self.max_num_detections
        elif (self.bin == DetectionsAcc.LABEL_BIN or
                self.bin == DetectionsAcc.LOCATION_BIN or
                self.bin == DetectionsAcc.LOCATION_LABEL_BIN):
            self.detection_len = 0
            for id, count in self.bin_counts.items():
                self.id_index_map[id] = self.detection_len
                self.detection_len += count
        else:
            raise ValueError("Invalid bin parameter")


        self.detections_tensor = torch.ones(
            (len(self.detections_list), self.detection_len, 7)
        )*float('inf')
        # self.detections_tensor[0:len(self.detections_list)//2] *= -1
        for i, detection in enumerate(self.detections_list):
            if detection is not None:
                if self.sort == DetectionsAcc.OBJECT_SORT:
                    detection_count = detection.size(0)
                elif self.sort == DetectionsAcc.CENTER_SORT:
                    detection_count = detection.size(0)
                    midy = (detection[:, 1]+detection[:, 3])/2
                    _, sort_idx = midy.sort(dim=0)
                    detection = detection[sort_idx]
                    midx = (detection[:, 0]+detection[:, 2])/2
                    _, sort_idx = midx.sort(dim=0)
                    detection = detection[sort_idx]

                if self.bin == DetectionsAcc.SINGLE_BIN:
                    self.detections_tensor[i, 0:detection_count] = detection
                elif (self.bin == DetectionsAcc.LABEL_BIN or
                        self.bin == DetectionsAcc.LOCATION_BIN or
                        self.bin == DetectionsAcc.LOCATION_LABEL_BIN):
                    if self.bin == DetectionsAcc.LABEL_BIN:
                        ids = detection[:, -1]
                        unique_ids = detection[:, -1].unique()
                    elif self.bin == DetectionsAcc.LOCATION_BIN:
                        midx = (detection[:, 0] + detection[:, 2]) / 2
                        midy = (detection[:, 1] + detection[:, 3]) / 2
                        xids = (midx / 416 * self.loc_bin_count).floor()
                        yids = (midy / 416 * self.loc_bin_count).floor()
                        ids = xids + yids * 10
                        unique_ids = ids.unique()
                    elif self.bin == DetectionsAcc.LOCATION_LABEL_BIN:
                        midx = (detection[:, 0] + detection[:, 2]) / 2
                        midy = (detection[:, 1] + detection[:, 3]) / 2
                        xids = (midx / 416 * self.loc_bin_count).floor()
                        yids = (midy / 416 * self.loc_bin_count).floor()
                        labels = detection[:, -1]
                        ids = xids + yids * 10 + labels * 100
                        unique_ids = ids.unique()

                    for id in unique_ids:
                        filtered_detection = detection[ids == id]
                        filtered_len = filtered_detection.size(0)
                        idx_st = self.id_index_map[id.cpu().item()]
                        self.detections_tensor[i, idx_st:idx_st+filtered_len]= filtered_detection





        self.detections_tensor, _ = self.detections_tensor.sort(dim=0)
    def median(self):
        result = self.detections_tensor[len(self.detections_list) // 2]
        return result
    def upper(self, alpha=.05):
        result = self.detections_tensor[int(len(self.detections_list)*(alpha))]
        return result
    def lower(self, alpha=.05):
        result = self.detections_tensor[int(len(self.detections_list)*(1-alpha))]
        return result
    def k(self, q):
        result = self.detections_tensor[q]
        return result
    def clear(self):
        self.detections_list = []
        self.max_num_detections = 0
        self.detections_tensor = None


def estimated_qu_ql(eps, sample_count, sigma, conf_thres = .99999):
    theo_perc_u = stats.norm.cdf(eps/sigma)
    theo_perc_l = stats.norm.cdf(-eps / sigma)

    q_u_u = sample_count + 1
    q_u_l = math.ceil(theo_perc_u*sample_count)
    q_l_u = math.floor(theo_perc_l*sample_count)
    q_l_l = 0
    q_u_final = q_u_u
    for q_u in range(q_u_l, q_u_u):
        conf = stats.binom.cdf(q_u-1, sample_count, theo_perc_u)
        if conf > conf_thres:
            q_u_final = q_u
            break

    q_l_final = q_l_l
    for q_l in range(q_l_u, q_l_l, -1):
        conf = 1-stats.binom.cdf(q_l-1, sample_count, theo_perc_l)
        if conf > conf_thres:
            q_l_final = q_l
            break

    return q_u_final, q_l_final



q_u, q_l = estimated_qu_ql(eps=0.05, sample_count=1000, sigma=0.12, conf_thres = .999)


class SmoothMedianNMS(nn.Module):
    def __init__(self, base_detector, sigma, accumulator):
        super(SmoothMedianNMS, self).__init__()
        self.base_detector = model
        self.sigma = sigma
        self.detection_acc = []

    def predict_range(self, x, n, batch_size, q_u, q_l, dn=False):

        input_imgs = x.repeat((batch_size, 1, 1, 1))
        for i in range(n//batch_size):
            # Get detections
            with torch.no_grad():
                out = input_imgs + torch.randn_like(input_imgs) * self.sigma
                out[out > 1] = 1
                out[out < 0] = 0
                if dn:
                  out = denoiser(out)
                detections = self.base_detector(out).squeeze()
                #print(detections.shape)
                if len(self.detection_acc) == 0:
                  self.detection_acc = detections
                else:
                  self.detection_acc = torch.concatenate([self.detection_acc, detections], axis=0)

        #self.detection_acc.tensorize()
        detections = [self.detection_acc.median()]
        #print(len(self.detection_acc))
        self.detection_acc = sorted(self.detection_acc)[::-1]
        #print(len(self.detection_acc), q_l, q_u)
        detections_l = [self.detection_acc[q_l]]
        detections_u = [self.detection_acc[q_u]]
        #self.detection_acc.clear()
        return detections, detections_u, detections_l



from tqdm import tqdm
import pandas as pd

df = pd.read_csv('res2/nms_koncept.csv')

dic = {}
for i in tqdm(range(len(df))):
    img = df.iloc[i]['path'].split('/')[-1][:-4]+'.jpg'
    print(df.iloc[i]['path'])
    print(img)
    if img not in dic:
        path = os.path.join('../../../../data/DIONE/work/Framework_Datasets/dataset/quality-sampled-datasets/koniq_sampled_MOS/1000_10_clusters', img)
        print(path)
        accumulator = DetectionsAcc()
        smoothed_model = SmoothMedianNMS(model, 0.12, accumulator)
        im = cv2.imread(path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype('float32') / 255.
        #im = cv2.resize(im, (256, 256))
        im = torch.from_numpy(im).to(device).permute(2, 0, 1).to(device)
        #im = torch.nn.functional.interpolate(im, (256, 256), mode='bicubic', antialias=True)
        detections, detections_l, detections_u = smoothed_model.predict_range(im, n=1000, batch_size=10, q_u=q_u, q_l=q_l, dn=False)
        print(detections, detections_l, detections_u)
        dic[img] = [detections, detections_l, detections_u]

df = pd.DataFrame([], columns=['path', 'median', 'lower_b', 'upper_b'])
df.to_csv('res2/clear_nms_koncept.csv', index=False)

for img in dic:
  df2 = pd.DataFrame([[img, dic[img][0], dic[img][1], dic[img][2]]], columns=['path', 'median', 'lower_b', 'upper_b'])
  df = pd.concat([df2, df])


df.to_csv('res2/clear_nms_koncept.csv', index=False)



df = pd.read_csv('res2/nms_dn_koncept.csv')

dic = {}
for i in tqdm(range(len(df))):
    img = df.iloc[i]['path'].split('/')[-1][:-4]+'.jpg'
    print(df.iloc[i]['path'])
    print(img)
    if img not in dic:
        path = os.path.join('../../../../data/DIONE/work/Framework_Datasets/dataset/quality-sampled-datasets/koniq_sampled_MOS/1000_10_clusters', img)
        #print(path)
        accumulator = DetectionsAcc()
        smoothed_model = SmoothMedianNMS(model, 0.12, accumulator)
        im = cv2.imread(path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype('float32') / 255.
        #im = cv2.resize(im, (256, 256))
        im = torch.from_numpy(im).to(device).permute(2, 0, 1).to(device)
        #im = torch.nn.functional.interpolate(im, (256, 256), mode='bicubic', antialias=True)
        detections, detections_l, detections_u = smoothed_model.predict_range(im, n=1000, batch_size=10, q_u=q_u, q_l=q_l, dn=True)
        #print(prediction, radius)
        dic[img] = [detections, detections_l, detections_u]

df = pd.DataFrame([], columns=['path', 'median', 'lower_b', 'upper_b'])
df.to_csv('res2/clear_nms_dn_koncept.csv', index=False)

for img in dic:
  df2 = pd.DataFrame([[img, dic[img][0], dic[img][1], dic[img][2]]], columns=['path', 'median', 'lower_b', 'upper_b'])
  df = pd.concat([df2, df])


df.to_csv('res2/clear_nms_dn_koncept.csv', index=False)
