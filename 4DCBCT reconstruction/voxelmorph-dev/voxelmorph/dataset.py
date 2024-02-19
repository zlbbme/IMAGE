import os
import torch
from os.path import join as pjoin
from scipy import misc

import matplotlib.pyplot as plt
import skimage.io as io
import torch.utils.data as data
import glob
import numpy as np
from torchvision.transforms import transforms as T
import torch.utils.data as data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class Nnet_Dataset(data.Dataset):

    def __init__(self, root_FDKImg, root_Prior, root_GT, HMIndex, SliceNum, transform=None, target_transform=None):
        self.root_FDKImg = root_FDKImg
        self.root_Prior = root_Prior
        self.root_GT = root_GT
        TrainingSet = []

        # for SetIndex in range(0, len(HMIndex)):
        for SetIndex in range(0, 1):

            for phase in range(1, 11):

                for sliceindex in range(0, SliceNum[SetIndex]):
                    new_degraded = glob.glob(
                        root_FDKImg + str(HMIndex[SetIndex]) + 'HM10395/Phase' + str(phase) + '/Processed' + str(
                            sliceindex + 1) + '.png')
                    new_prior = glob.glob(
                        root_Prior + str(HMIndex[SetIndex]) + 'HM10395/Prior/Prior' + str(sliceindex + 1) + '.png')
                    new_mask = glob.glob(
                        root_GT + str(HMIndex[SetIndex]) + 'HM10395/GT_Phase' + str(phase) + '/GT' + str(
                            sliceindex + 1) + '.png')

                    TrainingSet.append([new_degraded, new_prior, new_mask])

            self.TrainingSet = TrainingSet
            self.transform = transform
            self.target_transform = target_transform

    def __getitem__(self, index):

        x_path, prior_path, gt_path = self.TrainingSet[index]
        img_x = io.imread(x_path[0])  # image demension 512*512，value scale[0 255]
        img_prior = io.imread(prior_path[0])
        img_gt = io.imread(gt_path[0])

        shape = img_x.shape
        zeros = np.zeros((*shape, len(shape)))

        if self.transform is not None:
            img_x = self.transform(img_x)
            img_prior = self.transform(img_prior)
        if self.target_transform is not None:
            img_gt = self.target_transform(img_gt)
            zeros = self.target_transform(zeros)

        return (img_gt, img_x), (img_x, zeros)

    def __len__(self):
        return len(self.TrainingSet)


