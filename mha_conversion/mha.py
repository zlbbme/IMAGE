import cv2
import SimpleITK as sitk
import numpy as np
import os
#image = sitk.ReadImage('./ClinicalElektaDatasets/P1/CE_P1_Prior/CT_01.mha')
image = sitk.ReadImage('./ClinicalElektaDatasets/P1/CE_P1_T_01/FDKRecon/FDK4D_02.mha')
img_data = sitk.GetArrayFromImage(image)
print(img_data.shape) 
height = img_data.shape[0]
weight = img_data.shape[2]
channel = img_data.shape[1]
savepath = './imgtest'
#判断是否存在文件夹如果不存在则创建为文件夹
if not os.path.exists(savepath):
    os.makedirs(savepath)

for i in range(channel):
    img = np.zeros((height,weight), dtype=np.uint8)
    img = img_data[:,i,:]#*255
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    #顺时针旋转180度
    img = cv2.rotate(img, cv2.ROTATE_180)
    #重构为512*512
    img = cv2.resize(img,(512,512),interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(savepath+'/'+str(i+1)+'.png',img)
    #print('save image: ',i+1)
    