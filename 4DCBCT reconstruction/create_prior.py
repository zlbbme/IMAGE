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
            img[img<1] = 0 ;img[img>254] = 254
        elif file.endswith('.npy'):
            img = np.load(image_path).astype(np.uint16)
        #将图片数组添加到三维数组中
        img_phase[:,:,num] = img  #[512, 512, image_depth]

    assert img_phase.shape[2] == image_depth

    return img_phase 

def read_4D_image(input_patient_path):
    #获取input_patient_path文件夹下的所有文件夹名
    filelist = os.listdir(input_patient_path)
    #剔除掉文件夹名不包含数字的文件夹，只添加10个时相的图像
    filelist = [file for file in filelist if re.findall(r"\d+", file)] 
    #print(filelist)
    phase_num = 0
    for j ,phase in enumerate(filelist):
        if 'CBCT' in phase:

            #将图片的路径和文件名拼接起来
            phase_path = os.path.join(input_patient_path, phase)
            #打开图像并转换成数组
            img_phase = np.expand_dims(read_phase_image(phase_path),axis=0) 
            image_4D = np.concatenate((image_4D, img_phase), axis=0) if j!=0 else img_phase
            phase_num += 1
            print('phase%d has been read'%(j+1))

    #确定image_4D的第一维度是phase_num，不是的话报错
    assert image_4D.shape[0] == phase_num
        
    return image_4D

def consturct_4D_image(input_patient_path,intensity='AVG'):

    #读取4D图像
    image_4D = read_4D_image(input_patient_path)
    print(image_4D.shape)

    if intensity == 'AVG':
        print('AVG intensity Image has been selected')
        consturcted_image = np.mean(image_4D, axis=0)
    if intensity == 'MIP':
        print('MIP intensity Image has been selected')
        consturcted_image = np.max(image_4D, axis=0)
    print('4D image has been constructed')
    return consturcted_image

def creat_prior_imag(input_patient_path,intensity='AVG',output_type='png'):
    #构建4D图像
    consturcted_image = consturct_4D_image(input_patient_path,intensity)
    output_prior_path = input_patient_path+'/CBCTprior'+intensity
    #创建文件夹
    if not os.path.exists(output_prior_path):
        os.mkdir(output_prior_path)
    #将numpy数组转换为图片
    for j in range(consturcted_image.shape[2]):
        img_single = consturcted_image[:,:,j]
        #将png_avg转为位深度为8的灰度图
        if output_type == 'png':
            img_single = Image.fromarray(np.uint8(img_single))
            img_single.save(output_prior_path+'/'+str(j)+'.png')
        elif output_type == 'npy':
            img_single = np.array(img_single)
            np.save(output_prior_path+'/'+str(j)+'.npy',img_single)
  
    print('prior image has been saved in '+output_prior_path)

def batch_construct_4D_image(input_path,output_type):
    #获取input_path文件夹下的所有文件夹名
    patient_list = os.listdir(input_path)
    for patient in patient_list:

        input_patient_path = os.path.join(input_path, patient)
        creat_prior_imag(input_patient_path,intensity='AVG',output_type=output_type)
        creat_prior_imag(input_patient_path,intensity='MIP',output_type=output_type)

if __name__ == '__main__':
    input_path = r'E:\dataset\temp_png'
    batch_construct_4D_image(input_path,output_type='png')