import os
#只能使用指定显卡,0卡只能用于临床，数据训练只能用1，2，3卡
gpu_list = '0,1,2,3'
cuda = os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list   #需要设置在import torch 之前和import 使用了torch的库的前面  #此行代码把原来的1，2，3卡变成编号为0，1，2卡
device_num = [0,1,2,3]
import torch  ## the top-level pytorch package and tensor library
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pytorch_msssim import MS_SSIM, ms_ssim, SSIM, ssim

# common transforms for image processing
import torchvision 
from torchvision.transforms import transforms as T
import torch.utils.data as data
import matplotlib.pyplot as plt
from losses import CharbonnierLoss
import numpy as np
from tqdm import tqdm


## the Python debugger
torch.set_grad_enabled(True)
print( torch.__version__)
print( torchvision.__version__)
print( torch.cuda.device_count())
print( torch.cuda.is_available())

train_BATCH_NUM = 4
val_BATCH_NUM = 1
WORKER_NUM = 8
EPOCH = 200           #文章设置为50epoch结束
LEARNING_RATE = 1e-4  #修改！文章初始化设为1e-5 ，并且每5epoch调整至90%
MINI_BATCH = 50    #修改！文章设置为1000

## import dataset 
from TrainDataset_CSnet import *

## import model 
from model_newCSnet import *
x_transform = T.ToTensor()
y_transform = T.ToTensor() # normalize image value to [0 1]，归一化方法 直接除以255

# x_transform = T.Compose([
#     #T.ColorJitter(0.3, 0, 0, 0), #改变图像属性
#     T.ToTensor(),                #图像转换成张量，HWC格式转置成CHW，归一化到[0,1]
#     T.Normalize([0.1436],[0.2292])  #归一化到[-1,1]
# ])                 #修改！
# y_transform = T.Compose([
#     #T.ColorJitter(0.3, 0, 0, 0), #改变图像属性
#     T.ToTensor(),                #图像转换成张量，HWC格式转置成CHW，归一化到[0,1]
#     T.Normalize([0.1436],[0.2292])  #归一化到[-1,1]
# ])                 #修改！

#设置17个训练集的层数和前缀名
## set the index of training  dataset
SliceNum = [ 87, 66, 60, 79, 65, 71, 52, 60, 60, 67, 77, 92, 59, 67, 59, 51, 68 ]
## set the index of training  dataset
HMIndex = [ 100, 101, 102, 103, 104, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119 ]

train_dataset_4DCBCT = TrainDataset_CircleNnet3D(
         '/Data/SaveBibMip-SX/4DCBCT/'         #root_FDKImg
        ,'/Data/SaveBibMip-SX/4DCBCT/'         #root_Prior
		,'/Data/SaveBibMip-SX/4DCBCT/'         #root_GT
        ,HMIndex                      #前缀名
        ,SliceNum                     #断层层数     
        ,transform = x_transform     #训练集转换
        ,target_transform = y_transform
        )

print('The total trainingdata has', len(train_dataset_4DCBCT))

# split the dataset into training data and validation data
train_db, val_db = data.random_split(
        train_dataset_4DCBCT
        , [int(len(train_dataset_4DCBCT)*0.99),int(len(train_dataset_4DCBCT)*0.01)])
print('train:', len(train_db), 'validation:', len(val_db))

TrainDataLoader_SpatialCNN = data.DataLoader(
        train_db
        ,batch_size = train_BATCH_NUM
        ,shuffle = True
        ,num_workers = WORKER_NUM
        ,pin_memory= True  #修改，添加
        )

ValidDataLoader_SpatialCNN = data.DataLoader(
        val_db
        ,batch_size = val_BATCH_NUM
        ,shuffle = True
        ,num_workers = WORKER_NUM
        ,pin_memory= True  #修改，添加
        )

#指定显卡训练
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#model = CycNnet()
model = CSnet_Small_res()
#model = CSnet()

#修改！假分布式
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs")
    # dim = 0 [64, xxx] -> [32, ...], [32, ...] on 2GPUs
    model = nn.DataParallel( model, device_ids=[0,2,3] )

model = model.to(device)
## if use the intermediate trained model
# model_PATH ='./model_trained/CSnet_small_res_multidim_epoch20_1st.pth'
# model = (torch.load(model_PATH))     

#optimizer = optim.Adam( model.parameters(), lr = LEARNING_RATE )   #定义优化器 
#定义优化器为AdamW，学习率为1e-4，权重衰减为1e-4
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

scheduler_1 = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)   #可变学习率的调整

criterion_1 = torch.nn.MSELoss().cuda()  #定义损失函数  
criterion_2 = CharbonnierLoss().cuda()


MINI_Epoch = 0

## train loop
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  #可以通过每个参数组的求和得出参数量（计算可训练的参数if p.requires_grad）
print("Total_params: {}".format(pytorch_total_params))

for epoch in range(0,EPOCH):
    
    print("epoch={}, lr={}".format(epoch+1, optimizer.state_dict()['param_groups'][0]['lr']))
    
    running_loss = 0
    epoch_iterator = tqdm(TrainDataLoader_SpatialCNN, desc="Iteration(Epoch {})".format(epoch+1))
    #with tqdm(total=len(TrainDataLoader_SpatialCNN)) as t:

    for i,batch in enumerate( epoch_iterator, 0 ):
        
        img_seq_1, img_seq_2, img_seq_3, prior, labels = batch  
        #print('几个输入的维度',img_seq_1.shape,img_seq_2.shape,img_seq_3.shape,prior.shape,labels.shape)
        img_seq_1 = img_seq_1.to(device)
        img_seq_2 = img_seq_2.to(device)
        img_seq_3 = img_seq_3.to(device)
        prior = prior.to(device)

        labels = labels.to(device)
        #print(labels.shape)

        optimizer.zero_grad()  
        
        try:
            prediction = model( img_seq_1, img_seq_2, img_seq_3, prior )
            #print(prediction.shape)
        except RuntimeError as exception:
            if "out of memory" in str( exception ):
                print( "WARNING: out of memory" )
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                raise exception
        loss_1 = criterion_1( prediction,labels )    # MSELoss
        loss_2 = criterion_2( prediction,labels )    # CharbonnierLoss
        loss = loss_1 + loss_2
        #loss.backward(retain_graph=True)
        loss.backward()
        optimizer.step()    
        
        running_loss += loss.item()
        
        if i % MINI_BATCH == (MINI_BATCH-1):    #每个epoch中到一个MINI_BATCH就打印一次,根据train_BATCH_NUM=8，11400*0.9=10260，每个epoch中有10260/8=1282个MINI_batch，每50个MINI_batch打印一次，最多是batch1250
            print('[epoch %d batch %5d] loss: %.3f' %
                ( epoch+1, i+1, running_loss)
                    )
            
            running_loss = 0.0
            
            MINI_Epoch += 1 
    
    #### validation
    with torch.no_grad():
        running_loss_val = 0 ;runing_psnr=0;runing_ssim=0
        for j, batch_val in enumerate( ValidDataLoader_SpatialCNN, 0 ):

            images_val_1, images_val_2, images_val_3, prior_val, labels_val = batch_val  # [N, 1, H, W]
            
            images_val_1 = images_val_1.to(device)
            images_val_2 = images_val_2.to(device)
            images_val_3 = images_val_3.to(device)
            prior_val = prior_val.to(device)
            labels_val = labels_val.to(device)

            prediction_val = model(images_val_1, images_val_2, images_val_3, prior_val)
            #loss_val = criterion_1( prediction_val,labels_val )  +criterion_2(prediction_val,labels_val)  #MSE+CharbonnierLoss
            loss_val = criterion_2(prediction_val,labels_val) #CharbonnierLoss

            running_loss_val += loss_val.item()
        torch.cuda.empty_cache()
        print('[epoch %d batch %5d] Val_loss: %.3f' %
                        ( epoch+1, j+1, running_loss_val)
                        )
        
            
    scheduler_1.step()      #修改！添加
    PATH = './model_trained/'
    #判断路径是否存在，如不存在则创建一个文件夹
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    if epoch % 10 == 9:      
        PATH ='./model_trained/CSnet_small_res_multidim_epoch'+str(epoch+1)+'_5st.pth'    
        torch.save(model, PATH)  
         

