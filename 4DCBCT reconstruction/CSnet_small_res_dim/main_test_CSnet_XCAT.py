######## test XCATfemale512
import os
gpu_list = '0,1,2,3'
cuda = os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list   #需要设置在import torch 之前和import 使用了torch的库的前面  #此行代码把原来的1，2，3卡变成编号为0，1，2卡
import torch  ## the top-level pytorch package and tensor library
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


# from torchvision import transforms  # common transforms for image processing
from torchvision.transforms import transforms as T

from torch.utils.data import Dataset # an abstract class for representing a dataset
from torch.utils.data import DataLoader # wraps a dataset and provides access to the underlying data

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

import os
import torch
from os.path import join as pjoin
from scipy  import misc

import skimage.io as io ## skimage
import torch.utils.data as data
import glob

from torchvision.transforms import transforms as T
import torch.utils.data as data
from torch.utils.data import Dataset # an abstract class for representing a dataset
from torch.utils.data import DataLoader # wraps a dataset and provides access to the underlying data

## import CycN-Net model 
from model_newCSnet import *
from model_newCSnet import *

## import testing data
from TestingDataset_CSnet_XCAT import *

   
x_transform = T.ToTensor()
y_transform = T.ToTensor()
# #训练时这样操作了，测试也需要
# x_transform = T.Compose([
#     #T.ColorJitter(0.3, 0, 0, 0), #改变图像属性
#     T.ToTensor(),                #图像转换成张量，HWC格式转置成CHW，归一化到[0,1]
#     T.Normalize([0.1615],[0.2485])  #归一化到[-1,1]
# ])                 #修改！
# y_transform = T.Compose([
#     #T.ColorJitter(0.3, 0, 0, 0), #改变图像属性
#     T.ToTensor(),                #图像转换成张量，HWC格式转置成CHW，归一化到[0,1]
#     T.Normalize([0.1615],[0.2485])  #归一化到[-1,1]
# ])                 #修改！


SliceNum = 87   #test_dataset0=160  test_dataset_real=87 100HM10395=87

# dataset_4DCBCT = TestingDataset_XCAT(
#         './test_dataset0/'
#         ,'./test_dataset0/'
# 		,'./test_dataset0/'
#         ,SliceNum
#         ,transform = x_transform
#         ,target_transform = y_transform
#         )

dataset_4DCBCT = TestingDataset_XCAT(
        '/Data/SaveBibMip-SX/4DCBCT/100HM10395'   #原地址为：'./test_dataset0/'
        ,'./test_dataset_real/'
		,'./test_dataset_real/'
        ,SliceNum
        ,transform = x_transform
        ,target_transform = y_transform
        )

test_dataloader = DataLoader( dataset_4DCBCT, batch_size=1, shuffle=False, num_workers=0 )

## load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_PATH ='./model_trained/CSnet_small_res_multidim_epoch40_4st.pth'             #0

model_save = torch.load(model_PATH, map_location='cuda:1')

if isinstance(model_save,torch.nn.DataParallel):
		model_save = model_save.module   

# set the save path of processed image        
root_save = r'./Results_Real/CS_small_res_multidim_epoch40_4st/'

for i, batch in enumerate( test_dataloader, 0 ):
    
        img_seq_1, img_seq_2, img_seq_3, prior, labels = batch
        
        img_seq_1 = img_seq_1.to(device)
        img_seq_2 = img_seq_2.to(device)
        img_seq_3 = img_seq_3.to(device)
        prior = prior.to(device)
        labels = labels.to(device)
        
        PhaseIndex, SliceIndex = divmod( i, SliceNum )
        
        path = root_save+'/Phase'+str( PhaseIndex + 1 )

        isExists = os.path.exists(path)

        if not isExists:

            os.makedirs(path)
            print( path + 'create successfully!')
        
        with torch.no_grad():
            
            output = model_save(img_seq_1,img_seq_2,img_seq_3, prior)
            
        # output = output[0][0].mul(255).cpu().detach().squeeze().numpy()
        # heatmap = np.uint8( np.interp( output, (output.min(), output.max()), (0, 255)))
        # im = Image.fromarray(heatmap)
        # im.save( path + '/Processed' + str(SliceIndex+1) +'.png')
        output = output[0][0].mul(65536).cpu().detach().squeeze().numpy()       #转成16位
        heatmap = np.uint16( np.interp( output, (output.min(), output.max()), (0, 65536)))
        im = Image.fromarray(heatmap)
        im.save( path + '/Processed' + str(SliceIndex+1) +'.png')


print('processing finished!')
