import numpy as np
import os
#导入读取图片的包
from PIL import Image
import re

def read_phase_image(input_phase_path):
    #获取input_patient_path文件夹下的所有文件夹名
    filelist = os.listdir(input_phase_path)
    
    image_depth = len(filelist) ;image_high = 512 ;image_width = 512
    #创建一个空的三维数组，用于存储图片
    img_phase = np.zeros((image_high, image_width, image_depth))
    #遍历所有文件名
    for file in filelist:
        #读取file中的数字
        num = int(re.findall(r"\d+", file)[0])
        #print(num)
        #将图片的路径和文件名拼接起来
        image_path = os.path.join(input_phase_path, file)
        #打开图像并转换成数组
        if file.endswith('.png'):
            img = np.array(Image.open(image_path)).astype(np.uint8)
        elif file.endswith('.npy'):
            img = np.load(image_path).astype(np.uint16)
        #将图片数组添加到三维数组中
        img_phase[:,:,num-1] = img  #[512, 512, image_depth]

    assert img_phase.shape[2] == image_depth

    return img_phase 

def read_4D_image(input_patient_path):
    #获取input_patient_path文件夹下的所有文件夹名
    filelist = os.listdir(input_patient_path)
    #print(filelist)
    phase_num = len(filelist)
    for j ,phase in enumerate(filelist):
        #将图片的路径和文件名拼接起来
        phase_path = os.path.join(input_patient_path, phase)
        #打开图像并转换成数组
        img_phase = np.expand_dims(read_phase_image(phase_path),axis=0) 
        image_4D = np.concatenate((image_4D, img_phase), axis=0) if j!=0 else img_phase
        print('phase%d has been read'%(j+1))

    #确定image_4D的第一维度是phase_num，不是的话报错
    assert image_4D.shape[0] == phase_num
        
    return image_4D

def consturct_4D_image(input_patient_path,intensity='AVG'):

    #读取4D图像
    image_4D = read_4D_image(input_patient_path)
    print(image_4D.shape)
    #print(image_4D.shape)
    # #获取image_4D的第三维度的长度
    # image_depth = image_4D.shape[3]
     #如果intensity为AVG，则image_4D的第一个维度的平均值
    if intensity == 'AVG':
        consturcted_image = np.mean(image_4D, axis=0)
    if intensity == 'MIP':
        consturcted_image = np.max(image_4D, axis=0)
    print('4D image has been constructed')
    return consturcted_image

def creat_prior_imag(input_patient_path,intensity='AVG',output_type='png'):
    #构建4D图像
    consturcted_image = consturct_4D_image(input_patient_path,intensity)
    #创建文件夹
    if not os.path.exists(input_patient_path+'/prior'):
        os.mkdir(input_patient_path+'/prior')
    #将numpy数组转换为图片
    for j in range(consturcted_image.shape[2]):
        img_single = consturcted_image[:,:,j]
        #将png_avg转为位深度为8的灰度图
        if output_type == 'png':
            img_single = Image.fromarray(np.uint8(img_single))
            img_single.save(input_patient_path+'/prior/'+str(j+1)+'.png')
        elif output_type == 'npy':
            img_single = np.array(img_single)
            np.save(input_patient_path+'/prior/'+str(j+1)+'.npy',img_single)
  
    print('prior image has been saved in '+input_patient_path+'/prior')
if __name__ == '__main__':
    #read_phase_image('test_dataset0\Degraded\Phase1')
    #read_4D_image('test_dataset0\Degraded')
    #consturct_4D_image('test_dataset0\Degraded',intensity='AVG')
    creat_prior_imag('test_dataset0\Degraded',intensity='MIP',output_type='png')