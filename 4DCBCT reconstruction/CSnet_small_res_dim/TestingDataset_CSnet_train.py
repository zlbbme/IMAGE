import os
import torch
from os.path import join as pjoin
from scipy  import misc

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io ## skimage
import torch.utils.data as data
import glob

from torchvision.transforms import transforms as T
import torch.utils.data as data
from torch.utils.data import Dataset # an abstract class for representing a dataset
from torch.utils.data import DataLoader # wraps a dataset and provides access to the underlying data


class TestingDataset_XCAT(data.Dataset):
    
    def __init__( self, root_FDKImg, root_Prior, root_GT, SliceNum, transform = None, target_transform = None):
        
        self.root_FDKImg = root_FDKImg
        self.root_Prior = root_Prior
        self.root_GT = root_GT        

        
        PhaseSequence_5frames = [[9,10,1,2,3],[10,1,2,3,4],[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7],[4,5,6,7,8],[5,6,7,8,9],[6,7,8,9,10],[7,8,9,10,1],[8,9,10,1,2]]
        TrainingSet = []
          
            
        for phase in range(0,10):
            
            for sliceindex in range( 1, SliceNum+1 ):
                
                new_degraded_1 = glob.glob( root_FDKImg +'HM10395/DegradePhase'+str(PhaseSequence_5frames[phase][0])+'/Recons'+str(sliceindex)+'.png') 
                new_degraded_2 = glob.glob( root_FDKImg +'HM10395/DegradePhase'+str(PhaseSequence_5frames[phase][1])+'/Recons'+str(sliceindex)+'.png')    
                new_degraded_3 = glob.glob( root_FDKImg +'HM10395/DegradePhase'+str(PhaseSequence_5frames[phase][2])+'/Recons'+str(sliceindex)+'.png')
                new_degraded_4 = glob.glob( root_FDKImg +'HM10395/DegradePhase'+str(PhaseSequence_5frames[phase][3])+'/Recons'+str(sliceindex)+'.png')    
                new_degraded_5 = glob.glob( root_FDKImg +'HM10395/DegradePhase'+str(PhaseSequence_5frames[phase][4])+'/Recons'+str(sliceindex)+'.png')

                new_prior = glob.glob( root_Prior  + 'HM10395/Prior/Prior'+str(sliceindex)+'.png')     
                new_mask = glob.glob( root_GT +'HM10395/GT_Phase'+str(PhaseSequence_5frames[phase][2])+'/GT'+str(sliceindex)+'.png')


                aa = [new_degraded_1, new_degraded_2, new_degraded_3, new_degraded_4, new_degraded_5], [ new_prior ],[new_mask]
                TrainingSet.append(aa) 

            
        self.TrainingSet = TrainingSet
        self.transform = transform
        self.target_transform = target_transform
        
# to get an element from the dataset at a specific index location with the dataset     
    def __getitem__(self,index):  
        img_1 = np.zeros(( 512, 512,1 ))   #修改！
        img_2 = np.zeros(( 512, 512,1 ))
        img_3 = np.zeros(( 512, 512,1))
        img_4 = np.zeros(( 512, 512,1))
        img_5 = np.zeros(( 512, 512,1))
        img1_noise_seq = np.zeros((  512, 512 ,3))
        img2_noise_seq = np.zeros((  512, 512 ,3))  
        img3_noise_seq = np.zeros((  512, 512 ,3))
        img_prior_seq = np.zeros((512, 512,1))
        img_gt_seq = np.zeros(( 512, 512,1)) 
        #取同一位置的不同时相的图像，取上述元组中的第0个列表，原始图像,img_1 为单幅图像
        img_1 = io.imread(self.TrainingSet[index][0][0][0])   #[512,512]
        img_2 = io.imread(self.TrainingSet[index][0][1][0])
        img_3 = io.imread(self.TrainingSet[index][0][2][0])
        img_4 = io.imread(self.TrainingSet[index][0][3][0])
        img_5 = io.imread(self.TrainingSet[index][0][4][0])
        #相邻三个时相分别组合,array读取是就为[H,W,C]
        img1_noise_seq[:,:,0]=np.array(img_1)
        img1_noise_seq[:,:,1]=np.array(img_2)
        img1_noise_seq[:,:,2]=np.array(img_3)
        
        img2_noise_seq[:,:,0]=np.array(img_2)
        img2_noise_seq[:,:,1]=np.array(img_3)
        img2_noise_seq[:,:,2]=np.array(img_4)
        
        img3_noise_seq[:,:,0]=np.array(img_3)
        img3_noise_seq[:,:,1]=np.array(img_4)
        img3_noise_seq[:,:,2]=np.array(img_5)
    #取上述元组中的第1个列表，先验图像
        img_prior = io.imread(self.TrainingSet[index][1][0][0])
        img_prior_seq[:,:,0] = np.array(img_prior)
    #取上述元组中的第2个列表，GT图像
#        img_gt = io.imread(self.TrainingSet[index][2][0][0])
#        img_gt_seq[:,:,0] = np.array(img_gt)
        
        # img1_noise_seq = img1_noise_seq.transpose((1,2,0))   
        # img2_noise_seq = img2_noise_seq.transpose((1,2,0))  
        # img3_noise_seq = img3_noise_seq.transpose((1,2,0))  
        # img_prior_seq = img_prior_seq.transpose((1,2,0))  
        # img_gt_seq = img_gt_seq.transpose((1,2,0))

        if self.transform is not None:
#            img_seq_5frames = self.transform(img_seq_5frames).float().div(255)
            img1_noise_seq = self.transform(img1_noise_seq).float().div(255).unsqueeze(dim=0)
            img2_noise_seq = self.transform(img2_noise_seq).float().div(255).unsqueeze(dim=0)
            img3_noise_seq = self.transform(img3_noise_seq).float().div(255).unsqueeze(dim=0)
            img_prior_seq = self.transform(img_prior_seq).float().div(255)

        if self.target_transform is not None:
            img_gt_seq = self.target_transform(img_gt_seq).float().div(255)
            
        return img1_noise_seq, img2_noise_seq, img3_noise_seq, img_prior_seq, img_gt_seq #返回的是图片，或者tensor
#        return img_seq_5frames, img_prior_seq, img_gt_seq    
    
    def __len__(self):   # retures the length of the dataset
        return len(self.TrainingSet)
    
# SliceNum = 160    
# x_transform = T.ToTensor()
# y_transform = T.ToTensor()    

## example 
# dataset_4DCBCT = TestingDataset_XCAT(
#         'E:/Pytorch/4DCBCT_TrainingData/XCAT_female_512_slice/'
#         ,'E:/Pytorch/4DCBCT_TrainingData/XCAT_female_512_slice/'
# 		,'E:/Pytorch/4DCBCT_TrainingData/XCAT_female_512_slice/'
#         ,SliceNum
#         ,transform = x_transform
#         ,target_transform = y_transform
# )

# test_dataloader = DataLoader( dataset_4DCBCT, batch_size=1, shuffle=False, num_workers=0 )

# aa,bb,cc,dd,ee = next(iter(test_dataloader)) 

if __name__ == '__main__':
    x_transform = T.ToTensor()
    y_transform = T.ToTensor()

    dataset_4DCBCT = TestingDataset_XCAT(
        '/Data/SaveBibMip-SX/100'#'/Data/SaveBibMip-SX/4DCBCT/100HM10395'#'./test_dataset0'
        , '/Data/SaveBibMip-SX/100'
		, '/Data/SaveBibMip-SX/100'
        ,87
        ,transform = x_transform
        ,target_transform = y_transform
      )
    dataloader = data.DataLoader(dataset_4DCBCT, batch_size=1, shuffle=False, num_workers=0)

    aa,bb,cc,dd,ee = next(iter(dataloader))
    print('img1:',aa.shape , 'img2:',bb.shape , 'img3:',cc.shape ,'img_prior:',dd.shape , 'img_gt:',ee.shape)