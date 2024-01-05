import torch  
import torch.nn as nn
import torch.nn.functional as F
from ReSwin_dim import PriorSwinNet
from EEnet import EdgeEnhanceREC
def ExtractionBlock_3D(InChannel_1, OutChannel_1, Kernal_1_depth, Kernal_1 ):   #3D-Conv, PReLU
        DownBlock_3D = nn.Sequential(
        nn.Conv3d( InChannel_1, OutChannel_1, ( Kernal_1_depth, Kernal_1, Kernal_1 )),
        nn.PReLU()
        )
        return DownBlock_3D

def ExtractionBlock_3D_2(InChannel_2, OutChannel_2, Kernal_2 ):                 #2D-Conv, PReLU
        DownBlock_2D = nn.Sequential(
                nn.Conv2d( InChannel_2, OutChannel_2, Kernal_2 ),
                nn.PReLU()
                )
        return DownBlock_2D

def ExtractionBlock(InChannel_1, OutChannel_1, Kernal_1, InChannel_2, OutChannel_2, Kernal_2 ):  #Block/pBlock  to Max pooling and C
        DownBlock = nn.Sequential(
                nn.Conv2d( InChannel_1, OutChannel_1, Kernal_1 ),
                nn.PReLU(),
                nn.Conv2d( InChannel_2, OutChannel_2, Kernal_2 ),
                nn.PReLU()
                ) 
        return DownBlock

def ExpansionBlock(InChannel1, OutChannel1, Kernel1, InChannel2, OutChannel2, Kernel2):         #Tcon  ,两个转置卷积堆叠
        UpBlock = nn.Sequential(
                nn.ConvTranspose2d(in_channels= InChannel1, out_channels=OutChannel1, kernel_size= Kernel1 )
                ,nn.PReLU()
                ,nn.ConvTranspose2d(in_channels=InChannel2 , out_channels=OutChannel2, kernel_size= Kernel2 )
                ,nn.PReLU()
                )
        return UpBlock

def FinalConv(InChannel_1, OutChannel_1, Kernal_1, InChannel_2, OutChannel_2, Kernal_2, InChannel_3, OutChannel_3, Kernal_3):  #FinalConv
        finalconv = nn.Sequential(
                nn.Conv2d( InChannel_1, OutChannel_1, Kernal_1, padding=0 )
                ,nn.BatchNorm2d(OutChannel_1,track_running_stats=False) 
                ,nn.PReLU()
                ,nn.Conv2d( InChannel_2, OutChannel_2, Kernal_2, padding=1 )
                ,nn.BatchNorm2d(OutChannel_2,track_running_stats=False)
                ,nn.PReLU()
                ,nn.Conv2d( InChannel_3, OutChannel_3, Kernal_3, padding=0 )
        #            ,nn.BatchNorm2d(OutChannel_3,track_running_stats=False)
                ,nn.PReLU()
                )
        return finalconv

class encoder_single(nn.Module):
        def __init__(self) :
                super(encoder_single,self).__init__()  
                ## Encode
                self.conv_encode1 = ExtractionBlock_3D( 1, 4, 3, 5 ) ## (512-5)+1=508   Cycle phase images [1,1,3,512,512]>[1,4,1,508,508]
                self.conv_encode1_2 = ExtractionBlock_3D_2( 4, 4, 5 ) ## (508-5)+1=504  
                self.conv_encode2 = ExtractionBlock( 4, 8, 3, 8, 8, 3 ) ## (252-2-2)=248
                self.conv_encode3 = ExtractionBlock( 8, 16, 3, 16, 16, 3 ) ## (124-2-2)=120
                self.conv_encode4 = ExtractionBlock( 16, 32, 3, 32, 32, 3 ) ## (60-2-2)=56
                self.conv_encode5 = ExtractionBlock( 32, 64, 3, 64, 64, 3 ) ## (28-2-2)=24
                self.conv_encode6 = ExtractionBlock( 64, 128, 3, 128, 128, 3 ) ## （12-2-2）=8 
                self.conv_encode7 = ExtractionBlock( 128, 256, 3, 256, 256, 1 ) ## (4-2-0)=2
                self.conv_encode8= ExtractionBlock( 256, 256, 1, 256, 256, 1 ) ## 1
                self.pool = nn.MaxPool2d(2) #504/2=252

        def forward(self, image1):
                DownConv1_img1 = self.conv_encode1(image1) #504@4   #ExtractionBlock_3D  [B,1,3,512,512]->[B,4,1,508,508]
                temp = DownConv1_img1.squeeze(dim=2)        #dim = 2 为1 压缩  [B,4,508,508]
        #        feature_conv.append(temp)
                DownConv1_img1 = self.conv_encode1_2(temp)                #[B,4,504,504]
                DownConv1_pool_img1 = self.pool(DownConv1_img1)          #[B,4,252,252]
                DownConv2_img1 = self.conv_encode2(DownConv1_pool_img1)   #[B,8,248,248]
                DownConv2_pool_img1 = self.pool(DownConv2_img1)          #[B,8,124,124]
                DownConv3_img1 = self.conv_encode3( DownConv2_pool_img1 ) #[B,16,120,120]
                DownConv3_pool_img1 = self.pool( DownConv3_img1 )        #[B,16,60,60]
                DownConv4_img1 = self.conv_encode4(DownConv3_pool_img1)   #[B,32,56,56]
                DownConv4_pool_img1 = self.pool(DownConv4_img1)          #[B,32,28,28]   
                DownConv5_img1 = self.conv_encode5(DownConv4_pool_img1)   #[B,64,24,24]
                DownConv5_pool_img1 = self.pool(DownConv5_img1)          #[B,64,12,12]
                DownConv6_img1 = self.conv_encode6(DownConv5_pool_img1)   #[B,128,8,8]
                #DownConv6_pool_img1=self.pool(DownConv6_img1)            #[B,128,4,4]
                #DownConv7_img1 = self.conv_encode7(DownConv6_pool_img1)   #[B,256,2,2]
                #DownConv7_pool_img1=self.pool(DownConv7_img1)            #[1,256,1,1] 
                #DownConv8_img1 = self.conv_encode8(DownConv7_pool_img1)   #[1,256,1,1]

                del DownConv1_pool_img1
                del DownConv2_pool_img1
                del DownConv3_pool_img1
                del DownConv4_pool_img1
                del DownConv5_pool_img1
                #del DownConv6_pool_img1
                #del DownConv7_pool_img1
                return DownConv1_img1,DownConv2_img1,DownConv3_img1,DownConv4_img1,DownConv5_img1,DownConv6_img1 #,DownConv7_img1,DownConv8_img1

class encoder_prior(nn.Module):
        def __init__(self) :
                super(encoder_prior,self).__init__()
                ## Encode- Prior
                self.sw = PriorSwinNet(img_size=512,embed_dim=48,num_layers=6, extract_layers=[1,2,3,4,5,6])

        def forward(self,prior):
                ## Encode- Prior
                Prior_sw1,Prior_sw2,Prior_sw3,Prior_sw4,Prior_sw5,Prior_sw6 = self.sw(prior) #Prior_sw7,Prior_sw8
                return Prior_sw1,Prior_sw2,Prior_sw3,Prior_sw4,Prior_sw5,Prior_sw6#,Prior_DownConv7,Prior_DownConv8
#构建解码器的类
class decoder(nn.Module):
        def __init__(self) :
                super(decoder,self).__init__()
        ## Decode
                self.up = nn.Upsample(scale_factor=2, mode='nearest')   #nearest ->bilinear
                self.Tconv1 = ExpansionBlock( 1024, 512, 1, 512, 512, 3 ) ## 4*4@512
                self.Tconv2 = ExpansionBlock( 512, 256, 3, 256, 256, 3 ) ## 12*12@256
                self.Tconv3 = ExpansionBlock( 256, 128, 3, 128, 128, 3 ) ## 40*40@48
                self.Tconv4 = ExpansionBlock( 128, 64, 3, 64, 64, 3 ) ## 126@16
                self.Tconv5 = ExpansionBlock( 64, 32, 3, 32, 32, 3 ) ## 512*512@1
                self.Tconv6 = ExpansionBlock( 32, 16, 3, 16, 16, 3 ) ## 512*512@1
                self.Tconv7 = ExpansionBlock( 16, 16, 5, 16, 16, 5) ## 512*512@1
                self.finalconv  = FinalConv( 16, 16, 1, 16, 8, 3, 8, 1, 1 ) 
                self.edge = EdgeEnhanceREC()

        def forward(self,img1,img2,img3,Prior):
                #Decode
                # temp = torch.cat((Prior[7],img1[7], img2[7],img3[7]), dim =1) #1@256*4
                # up1 = self.up(temp) # 2@256*4
                # temp = torch.cat((Prior[6],img1[6],img2[6], img3[6]), dim =1 ) #2@256*4
                # Tconv_1 = self.Tconv1(up1+temp)  #4@512
                # up2 = self.up(Tconv_1) #8@512

                temp = torch.cat((Prior[5],img1[5],img2[5],img3[5]), dim =1 )  #8@128*4
                #Tconv_2 = self.Tconv2(up2+temp)  #12@256
                Tconv_2 = self.Tconv2(temp)  #12@256

                up3 = self.up(Tconv_2) #24@256
                temp = torch.cat((Prior[4],img1[4],img2[4],img3[4]), dim =1 ) #24@64*4
                Tconv_3 = self.Tconv3(up3+temp)  #28@128
                
                up4 = self.up(Tconv_3) #56@128
                temp = torch.cat((Prior[3],img1[3],img2[3],img3[3]), dim =1 ) #56@32*4
                Tconv_4 = self.Tconv4(up4+temp)  #60@64
                
                up5 = self.up(Tconv_4) #120@64
                temp = torch.cat((Prior[2],img1[2],img2[2],img3[2]), dim =1 ) #120@16*4
                Tconv_5 = self.Tconv5(up5+temp)  #124@32
                
                up6 = self.up(Tconv_5) #248@32
                temp = torch.cat((Prior[1],img1[1],img2[1],img3[1]), dim =1 )  #248@8*3
                Tconv_6 = self.Tconv6(up6+temp)  #252@16
                
                up7 = self.up(Tconv_6) #504@16
                temp = torch.cat((Prior[0],img1[0],img2[0],img3[0]), dim =1 ) # 504@4*4
                Tconv_7 = self.Tconv7(up7+temp)  #508@16
                        
                out = self.finalconv(Tconv_7) # 508@16 --> 512@8 --> 512@1
                out = self.edge(out)
                del temp,up3,up4,up5,up6,up7,Tconv_2,Tconv_3,Tconv_4,Tconv_5,Tconv_6,Tconv_7
                return out

#卷积后图像大小计算公式：N = （W-F+2P）/S +1  ;W为输入大小，N为输出大小，F为卷积大小，P为默认为0，S默认为1
class CSnet_Small_res(nn.Module):

    def __init__(self,depths = 8):             #depth = 6
        super(CSnet_Small_res, self).__init__()        
        self.encoder_phase = encoder_single()
        self.encoder_prior = encoder_prior()
        self.decoder =decoder()
        
    def forward(self, image1,image2,image3, Prior):
## Encode 
# image_seq_1
        img1_encode = self.encoder_phase(image1)
# image_seq_2
        img2_encode = self.encoder_phase(image2)
# image_seq_3
        img3_encode = self.encoder_phase(image3)
## Encode- Prior
        Prior_encode = self.encoder_prior(Prior)
#Decode
        out = self.decoder(img1_encode,img2_encode,img3_encode,Prior_encode)
        del img1_encode,img2_encode,img3_encode,Prior_encode
        return out
    
if __name__ =='__main__':
        #指定使用显卡
        import os
        gpu_list = '0,1,2,3'
        cuda = os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list   
        device_num = [0,1,2,3]

        from torchvision.transforms import transforms as T
        from TrainDataset_CSnet import *	
        model = CSnet_Small_res()
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  #可以通过每个参数组的求和得出参数量（计算可训练的参数if p.requires_grad）
        print("Total_params: {}".format(pytorch_total_params))
        x_transform = T.ToTensor()
        y_transform = T.ToTensor()
                
        dataset_4DCBCT = TrainDataset_CircleNnet3D(
                '/Data/SaveBibMip-SX/4DCBCT/'         #root_FDKImg
                ,'/Data/SaveBibMip-SX/4DCBCT/'        #root_Prior
                ,'/Data/SaveBibMip-SX/4DCBCT/'        #root_GT
                ,[101]
                ,[66]
                , transform = x_transform
                , target_transform = y_transform
                )
        dataloader = data.DataLoader(dataset_4DCBCT, batch_size=1, shuffle=True, num_workers=0)

        aa,bb,cc,dd,ee = next(iter(dataloader)) # aa,bb,cc均为[1,3,512,512]  dd为[1,512,512]
       
        if cuda:
                aa = aa.cuda()
                bb = bb.cuda()
                cc = cc.cuda()
                dd = dd.cuda()
                ee = ee.cuda()
                model = model.cuda()
                model = nn.DataParallel(model, device_ids=device_num)
        else:
                print('no cuda')
        prediction = model( aa, bb, cc, dd )
        print('label',ee.shape)
        print('prediction',prediction.shape)

