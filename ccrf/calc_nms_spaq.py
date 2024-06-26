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
        if len(image.shape) > 3:
          res = []
          for i in range(image.shape[0]):
            patches = self.loader.generate_patches_custom(image[i], 224).to(self.device)
            res.append(self.model(patches).mean().item())
          return torch.from_numpy(np.array(res))
        patches = self.loader.generate_patches_custom(image, 224).to(self.device)
        return self.model(patches).mean()

model = MetricModel(device=device)

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

df = pd.DataFrame([], columns=['path', 'median', 'lower_b', 'upper_b'])
df.to_csv(f'res2/nms_spaq.csv', index=False)
df = pd.DataFrame([], columns=['path', 'median', 'lower_b', 'upper_b'])
df.to_csv(f'res2/nms_dn_spaq.csv', index=False)


paths = []
attacks = ['cadv', 'grad-est', 'korhonen-et-al', 'onepixel', 'patch-rs', 'ssah', 'uap',
'cnn-attack', 'ifgsm', 'madc', 'parsimonious', 'square-attack', 'stadv', 'zhang-et-al-dists']
presets = ['preset_0', 'preset_1', 'preset_2']
for preset in presets:
        for attack in attacks:
                path = os.path.join('../../../../data/DIONE/work/Framework_Datasets/dataset/attacked-dataset/no-defence/', preset, attack, 'spaq')
                files = sorted(os.listdir(path))[:10]
                for i in range(10):
                        paths.append(os.path.join(path, files[i]))


from tqdm import tqdm
import pandas as pd


for path in tqdm(paths):
  accumulator = DetectionsAcc()
  smoothed_model = SmoothMedianNMS(model, 0.12, accumulator)
  im = cv2.imread(path)
  im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype('float32') / 255.
  #im = cv2.resize(im, (256, 256))
  im = torch.from_numpy(im).to(device).permute(2, 0, 1).to(device)
  #im = torch.nn.functional.interpolate(im, (256, 256), mode='bicubic', antialias=True)
  detections, detections_l, detections_u = smoothed_model.predict_range(im, n=1000, batch_size=10, q_u=q_u, q_l=q_l, dn=False)
  df = pd.read_csv(f'res2/nms_spaq.csv')
  df2 = pd.DataFrame([[path, detections[0].item(), detections_l[0].item(), detections_u[0].item()]], columns=['path', 'median', 'lower_b', 'upper_b'])
  df = pd.concat([df2, df])
  df.to_csv(f'res2/nms_spaq.csv', index=False)

  accumulator = DetectionsAcc()
  smoothed_model = SmoothMedianNMS(model, 0.12, accumulator)
  im = cv2.imread(path)
  im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype('float32') / 255.
  #im = cv2.resize(im, (256, 256))
  im = torch.from_numpy(im).to(device).permute(2, 0, 1).to(device)
  #im = torch.nn.functional.interpolate(im, (256, 256), mode='bicubic', antialias=True)
  detections, detections_l, detections_u = smoothed_model.predict_range(im, n=1000, batch_size=10, q_u=q_u, q_l=q_l, dn=True)
  df = pd.read_csv(f'res2/nms_dn_spaq.csv')
  df2 = pd.DataFrame([[path, detections[0].item(), detections_l[0].item(), detections_u[0].item()]], columns=['path', 'median', 'lower_b', 'upper_b'])
  df = pd.concat([df2, df])
  df.to_csv(f'res2/nms_dn_spaq.csv', index=False)
