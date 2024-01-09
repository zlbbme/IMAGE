gdth_file = 'segmentation_metrics-master/img/l8_mask.nii.gz'  # ground truth image full path
pred_file = 'segmentation_metrics-master/img/l8_seg.nii.gz'  # prediction image full path

#导入包
import os
import nibabel as nib
import numpy as np

#计算dice系数
def dice_coefficient(gdth_file, pred_file):
    gdth = nib.load(gdth_file).get_data()
    pred = nib.load(pred_file).get_data()
    gdth = gdth.flatten()
    pred = pred.flatten()
    dice = 2 * np.sum(gdth * pred) / (np.sum(gdth) + np.sum(pred))
    return dice

dice= dice_coefficient(gdth_file, pred_file)
print(dice)

#通过pytorch计算dice系数
import torch
def dice_coefficient_pytorch(gdth_file, pred_file):
    gdth = nib.load(gdth_file).get_data()
    pred = nib.load(pred_file).get_data()
    gdth = torch.from_numpy(gdth)
    pred = torch.from_numpy(pred)
    gdth = gdth.reshape(-1)
    pred = pred.reshape(-1)
    dice = 2 * torch.sum(gdth * pred) / (torch.sum(gdth) + torch.sum(pred))
    return dice

dice_pytorch= dice_coefficient_pytorch(gdth_file, pred_file)
print(dice_pytorch)