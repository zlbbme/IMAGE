import nibabel as nib
import os
import numpy as np
import cv2

def nii_max_min(nii_path):
    max_list = []; min_list = [] ;len_dicom = 0
    #如果mha_path是文件
    if os.path.isfile(nii_path):
        # 读取mha文件
        nii_img = nib.load(nii_path)
        nii_data = nii_img.get_fdata()
        max_list.append(nii_data.max())
        min_list.append(nii_data.min())
        len_dicom = 1

    if os.path.isdir(nii_path):
    #遍历文件夹下的所有文件，获取后缀为.DCM的文件
        for root, dirs, files in os.walk(nii_path):
            for file in files:
                if file.endswith(".nii"):
                    len_dicom += 1
                    nii_path = os.path.join(root, file)
                    print(nii_path)
                # 读取 DICOM 文件
                nii_img = nib.load(nii_path)
                nii_data = nii_img.get_fdata()
                max_list.append(nii_data.max())
                min_list.append(nii_data.min())
                
    if np.min(max_list) > 500:
        max_CT_num = np.min(max_list) ; min_CT_num = np.max(min_list)   #获取最大的CT值为最大值列表的最小值，最小的CT值为最小值列表的最大值，压缩可用灰度区间
    else:
        max_CT_num = np.max(max_list) ; min_CT_num = np.min(min_list)
    return min_CT_num, max_CT_num, len_dicom
#定义nii转png函数
def convert_nii_to_png(nii_file,png_path):
    if not os.path.exists(png_path):
        os.makedirs(png_path)
    #读取nii文件
    nii_img = nib.load(nii_file)
    nii_data = nii_img.get_fdata()
    #获取nii文件的维度
    nii_shape = nii_data.shape
    #遍历nii文件的每一张图片
    for i in range(nii_shape[2]):
        #获取nii文件中的每一张图片
        nii_array = nii_data[:,:,i]
        #归一化
        nii_array = (nii_array - np.min(nii_array)) / (np.max(nii_array) - np.min(nii_array)) * 255
        #将nii文件中的每一张图片转换为png格式
        cv2.imwrite(os.path.join(png_path,str(i)+'.png'),nii_array)

if __name__ == '__main__':
    nii_path = r"E:\dataset\temp_dicom\100HM10395\nii\CTp1.nii"
    min, max ,len= nii_max_min(nii_path)
    print(max, min)
    png_path = nii_path.replace('.nii', '')
    convert_nii_to_png(nii_path, png_path)