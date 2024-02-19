# Testing Dataset XCAT phantom

import os
import torch
from os.path import join as pjoin
from scipy import misc

import matplotlib.pyplot as plt
import skimage.io as io  ## skimage
import torch.utils.data as data
import glob
import numpy as np
from torchvision.transforms import transforms as T
import torch.utils.data as data
from torch.utils.data import Dataset  # an abstract class for representing a dataset
from torch.utils.data import DataLoader  # wraps a dataset and provides access to the underlying data


class Nnet_Dataset(data.Dataset):

    def __init__(self, root_FDKImg, root_Prior, root_GT, SliceNum, transform=None, target_transform=None):

        self.root_FDKImg = root_FDKImg
        self.root_Prior = root_Prior
        self.root_GT = root_GT
        #        seeds=[ 101, 102, 103, 104, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119 ]
        TestingSet = []
        #        sliceNum = [66,60,79,65,71,52,60,60,67,77,92,59,67,59,51,68]

        #        for i,seeds_index in enumerate(seeds,0):

        for phase in range(1, 11):

            for sliceindex in range(1, SliceNum + 1):
                ct = glob.glob(root_GT + 'GT_Phase' + str(phase) + '/GT' + str(sliceindex) + '.png')
                cbct = glob.glob(root_FDKImg + 'Phase' + str(phase) + '/Processed' + str(sliceindex) + '.png')

                TestingSet.append([ct, cbct])

        self.TestingSet = TestingSet
        self.transform = transform
        self.target_transform = target_transform

    # to get an element from the dataset at a specific index location with the dataset
    def __getitem__(self, index):

        ct_path, cbct_path = self.TestingSet[index]
        img_ct = io.imread(ct_path[0])
        img_cbct = io.imread(cbct_path[0])

        shape = img_cbct.shape
        zeros = np.zeros((*shape, len(shape)))

        if self.transform is not None:
            img_ct = self.transform(img_ct)
            img_cbct = self.transform(img_cbct)
            zeros = self.target_transform(zeros)

        return (img_cbct, img_ct), (img_ct, zeros)

    def __len__(self):  # retures the length of the dataset
        return len(self.TestingSet)