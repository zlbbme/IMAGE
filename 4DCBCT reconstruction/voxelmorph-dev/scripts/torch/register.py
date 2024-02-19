#!/usr/bin/env python

"""
Example script to register two volumes with VoxelMorph models.

Please make sure to use trained models appropriately. Let's say we have a model trained to register 
a scan (moving) to an atlas (fixed). To register a scan to the atlas and save the warp field, run:

    register.py --moving moving.nii.gz --fixed fixed.nii.gz --model model.pt 
        --moved moved.nii.gz --warp warp.nii.gz

The source and target input images are expected to be affinely registered.

If you use this code, please cite the following, and read function docs for further info/citations
    VoxelMorph: A Learning Framework for Deformable Medical Image Registration 
    G. Balakrishnan, A. Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. 
    IEEE TMI: Transactions on Medical Imaging. 38(8). pp 1788-1800. 2019. 

    or

    Unsupervised Learning for Probabilistic Diffeomorphic Registration for Images and Surfaces
    A.V. Dalca, G. Balakrishnan, J. Guttag, M.R. Sabuncu. 
    MedIA: Medical Image Analysis. (57). pp 226-236, 2019 

Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in 
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or 
implied. See the License for the specific language governing permissions and limitations under 
the License.
"""

import os
import argparse
import neurite as ne
# third party
import numpy as np
import nibabel as nib
from torch import tensor
from PIL import Image
import cv2
import torch.nn as nn
import torch.nn.functional as F

import skimage.io as io

# import voxelmorph with pytorch backend
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm
from voxelmorph.dataset import *

# parse commandline args
parser = argparse.ArgumentParser()
parser.add_argument('--moving', default='moving', help='moving image (source) filename')
parser.add_argument('--fixed', default='fixed', help='fixed image (target) filename')
parser.add_argument('--moved', default='moved', help='warped image output filename')
parser.add_argument('--model', default='/home/wuweihang/medical image/voxelmorph-dev/models/0380.pt', help='pytorch model for nonlinear registration')
parser.add_argument('--warp', help='output warp deformation filename')
parser.add_argument('-g', '--gpu', default='1', help='GPU number(s) - if not supplied, CPU is used')
parser.add_argument('--multichannel', default=True, action='store_true',
                    help='specify that data has multiple channels')
args = parser.parse_args()

# device handling
device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def Laplacian(x):
    kernel = torch.FloatTensor([
        [[[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]]]
    ])
    b = torch.FloatTensor([0.])
    weight = nn.Parameter(data=kernel, requires_grad=False)
    b = nn.Parameter(data=b, requires_grad=False)
    out = F.conv2d(x, weight, b, 1, 1)
    return out

# load and set up model
model = vxm.networks.VxmDense.load(args.model, device)

# SliceNum = 160
#
# # instance of dataset
# x_transform = T.ToTensor()
# y_transform = T.ToTensor()
#
# train_dataset_4DCBCT = Nnet_Dataset(
#         '/home/wuweihang/medical image/N-Net_and_CycNet-master/CycN-Net/Results/CycNnet_Result_XCATfemale/',
#         '/home/wuweihang/Dataset/CBCT/N-NetDataset/test_dataset/',
#         '/home/wuweihang/Dataset/CBCT/N-NetDataset/test_dataset/GT/',
#         SliceNum, transform=x_transform, target_transform=y_transform)
#
# dataloader = DataLoader(train_dataset_4DCBCT, batch_size=1, shuffle=False, num_workers=4)

root_save = r'/home/wuweihang/medical image/voxelmorph-dev/results/'

# set the index of training  dataset
HMIndex = [100, 101, 102, 103, 104, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119]
# count the  slice number for each set of dataset
SliceNum = [87, 66, 60, 79, 65, 71, 52, 60, 60, 67, 77, 92, 59, 67, 59, 51, 68]

# instance of dataset
x_transform = T.ToTensor()
y_transform = T.ToTensor()

train_dataset_4DCBCT = Nnet_Dataset(
        '/home/wuweihang/medical image/N-Net_and_CycNet-master/CycN-Net/Results/train/',
    '/home/wuweihang/Dataset/CBCT/N-NetDataset/training_dataset/',
    '/home/wuweihang/Dataset/CBCT/N-NetDataset/training_dataset/', HMIndex, SliceNum
        , transform=x_transform
        , target_transform=y_transform
        )

dataloader = DataLoader(train_dataset_4DCBCT, batch_size=1, shuffle=False, num_workers=4)

temp = 0
j = 1
for i, batch in enumerate(dataloader, 0):
    inputs, y_true = batch

    # PatientIndex = HMIndex[0]
    # PhaseIndex, SliceIndex = divmod(i, SliceNum[0])
    #
    # if sum(SliceNum[0:j]) * 10 <= i < sum(SliceNum[0:j + 1]) * 10:
    #     temp = sum(SliceNum[0:j]) * 10
    #     PatientIndex = HMIndex[j]
    #     PhaseIndex, SliceIndex = divmod(i - temp, SliceNum[j])
    # elif i >= sum(SliceNum[0:j + 1]) * 10:
    #     j = j + 1
    #     temp = sum(SliceNum[0:j]) * 10
    #     PatientIndex = HMIndex[j]
    #     PhaseIndex, SliceIndex = divmod(i - temp, SliceNum[j])

    # path = root_save + str(PatientIndex) + 'HM10395/Phase' + str(PhaseIndex + 1)
    # isExists = os.path.exists(path)
    # if not isExists:
    #     os.makedirs(path)
    #     print(path + 'create successfully!')

    pre = model(*inputs)
    # cbct = inputs[1]
    # cbct = cbct[0][0].cpu().detach().squeeze().numpy()
    # cbct[cbct < 0] = 0
    # temp = np.zeros((512, 512, 3))
    # temp[:, :, 0] = cbct
    # temp[:, :, 1] = cbct
    # temp[:, :, 2] = cbct
    # cbct = temp
    #
    # image = cv2.imread('/home/wuweihang/medical image/voxelmorph-dev/data/100HM395_GT_label/phase' +
    #                    str(PhaseIndex + 1) + '_GTV_label/100_phase' + str(PhaseIndex + 1) + '_GTV' + str(SliceIndex+28) + '.png',
    #                    flags=cv2.IMREAD_GRAYSCALE)
    # image = torch.unsqueeze(x_transform(image), 0)
    # tansformer = vxm.layers.SpatialTransformer((512, 512))
    # label = tansformer(image, pre[1])
    # label = Laplacian(label)
    # label = label[0][0].cpu().detach().squeeze().numpy()
    # if np.max(label) != 0:
    #     label = label/np.max(label)
    # label[label < 0] = 0
    #
    # outcome = cbct
    # outcome[:, :, 0] = cbct[:, :, 0] + label
    # heatmap = np.uint8(np.interp(outcome, (outcome.min(), outcome.max()), (0, 255)))
    # im = Image.fromarray(heatmap)
    # im.save(path + '/Processed' + str(SliceIndex+1) +'.png')

    images = [img[0, 0, :, :] for img in inputs + list(pre)]
    images = [img.detach().numpy() for img in images]
    titles = ['moving', 'fixed', 'moved', 'flow']
    ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True)







