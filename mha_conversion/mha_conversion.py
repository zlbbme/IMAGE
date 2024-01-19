import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pydicom

def convert_mha_to_png(mha_file, path_png):
    image = sitk.ReadImage(mha_file)
    img_data = sitk.GetArrayFromImage(image)
    print(img_data.shape)
    depths = img_data.shape[1]
    height = img_data.shape[0]
    weight = img_data.shape[2]
    
    
    #判断是否存在文件夹如果不存在则创建为文件夹
    if not os.path.exists(path_png):
        os.makedirs(path_png)

    for i in range(depths):
        img = np.zeros((height,weight), dtype=np.uint8)
        img = img_data[:,i,:]
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        #镜面翻转
        img = cv2.flip(img, 1)
        #顺时针旋转180度
        img = cv2.rotate(img, cv2.ROTATE_180)
        #重构为512*512
        img = cv2.resize(img,(512,512),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(path_png+'/'+str(i+1)+'.png',img)


def convert_mha_to_npy(mha_file, npy_path):
    image = sitk.ReadImage(mha_file)
    img_data = sitk.GetArrayFromImage(image)
    print(img_data.shape)
    height = img_data.shape[0]
    weight = img_data.shape[2]
    channel = img_data.shape[1]
    
    #判断是否存在文件夹如果不存在则创建为文件夹
    if not os.path.exists(npy_path):
        os.makedirs(npy_path)

    for i in range(channel):
        img = np.zeros((height,weight), dtype=np.float32)
        img = img_data[:,i,:]
        #顺时针旋转180度
        img = cv2.rotate(img, cv2.ROTATE_180)
        #重构为512*512
        img = cv2.resize(img,(512,512),interpolation=cv2.INTER_CUBIC)
        #保存为mha文件
        np.save(npy_path+'/'+str(i+1)+'.npy',img)

def convert_mha_dicom(mha_file, dicom_path,dcm_file):
    # Read the existing DICOM file
    ds = pydicom.dcmread(dcm_file, force=True)
    # Read the original mha file
    image = sitk.ReadImage(mha_file)
    img_data = sitk.GetArrayFromImage(image)
    #输出img_data的类型
    print(img_data.dtype)
    height = img_data.shape[0]
    width = img_data.shape[2]
    channel = img_data.shape[1]
    if not os.path.exists(dicom_path):
        os.makedirs(dicom_path)

    for i in range(channel):
        img = np.zeros((height,width), dtype=np.int16)
        img = img_data[:,i,:]
        print(img.shape)
if __name__ == '__main__':
    #convert_mha_to_npy('CT_01.mha', './npy')
    #convert_mha_to_png('CT_01.mha','./png')
    # img_test = sitk.ReadImage('CT_01.mha')        #[x,y,z]
    # img_data = sitk.GetArrayFromImage(img_test)   #[z,y,x]
    # print(img_data.shape)
    # convert_mha_direction('CT_01.mha', 'CT_01_out.mha')
    # img_test2 = sitk.ReadImage('CT_01_out.mha')        #[x,y,z]
    # img_data2 = sitk.GetArrayFromImage(img_test2)   #[z,y,x]
    # print(img_data2.shape)
    #convert_mha_to_png('CT_01_out.mha','./png2')

    #convert_mha_to_png('CT_01.mha','./png')
    convert_mha_dicom('CT_01.mha','./dcm','2023021310_CT20_image00000.DCM')
        #获取dicom文件的tag
    #ds = sitk.ReadImage('./dcm/1.dcm')
    # #打印所有tag
    #print(ds)

    #print(spacing,origin,direction)