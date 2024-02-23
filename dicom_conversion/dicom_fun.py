import numpy as np
import matplotlib.pyplot as plt

import pydicom
from dicom_conversion import *

npy_data = np.load(r'\\192.168.202.30\FtpWorkDir\SaveBibMip-SX\eva_data\XCAT\npy\Old_result\Phase3\Processed15.npy')
# print(npy_data.min(),npy_data.max())
# # 使用matplotlib进行可视化
plt.imshow(npy_data,'gray')
plt.show()

# dicom_path = r'E:\dataset\Clinic_data\2021121308\CBCTAVG'

# dicom_files = os.listdir(dicom_path)

# for dicom_dile in dicom_files:
#     dicom_file_path = os.path.join(dicom_path,dicom_dile)
#     ds =pydicom.dcmread(dicom_file_path,force=True)
#     print(ds.InstanceNumber)
#     if ds.InstanceNumber == 75:
#         print('ok')
#         break

#CBCT的 instance number 是从0-74
#CT的 instance number 是从1-75
    

import pydicom
import matplotlib.pyplot as plt

def plot_dicom_histogram(dicom_path):
    #获取dicom文件路径
    dicom_files = os.listdir(dicom_path)
    #遍历dicom文件
    for dicom_file in dicom_files:
        #拼接dicom文件路径
        dicom_file_path = os.path.join(dicom_path,dicom_file)
        #读取dicom文件
        ds = pydicom.dcmread(dicom_file_path,force=True)
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        #获取instance number
        instance_number = ds.InstanceNumber
        #如果instance number为75，则打印文件路径
        #if instance_number == 50:
        print(dicom_file_path)
        #获取dicom文件的像素值
        pixel_array = ds.pixel_array
        #绘制直方图
        plt.hist(pixel_array.flatten(), bins=80, color='c')
        plt.xlabel("Hounsfield Units (HU)")
        plt.ylabel("Frequency")
        plt.show()
        #break
#定义绘制png文件直方图的函数
def plot_png_histogram(png_path):
    #获取png文件路径
    png_files = os.listdir(png_path)
    #遍历png文件
    for png_file in png_files:
        #拼接png文件路径
        png_file_path = os.path.join(png_path,png_file)
        #读取png文件
        png_data = plt.imread(png_file_path)*255    #plt.imread()读取的是0-1之间的数，*255转换为0-255之间的数
        #绘制直方图
        plt.hist(png_data.flatten(), bins=80, color='c')
        plt.xlabel("Hounsfield Units (HU)")
        plt.ylabel("Frequency")
        plt.show()
        break

#定义绘制npy文件直方图的函数
def plot_npy_histogram(npy_path):
    #获取npy文件路径
    npy_files = os.listdir(npy_path)
    #遍历npy文件
    for npy_file in npy_files:
        #拼接npy文件路径
        npy_file_path = os.path.join(npy_path,npy_file)
        #读取npy文件
        npy_data = np.load(npy_file_path)
        #绘制直方图
        plt.hist(npy_data.flatten(), bins=80, color='c')
        plt.xlabel("Hounsfield Units (HU)")
        plt.ylabel("Frequency")
        plt.show()
        # break
#定义绘制mha文件直方图的函数
def plot_mha_histogram(mha_path):
    
    #读取mha文件
    mha_data = sitk.ReadImage(mha_path)
    #获取mha文件的像素值
    mha_data = sitk.GetArrayFromImage(mha_data).transpose((2, 1, 0))
    #绘制直方图
    plt.hist(mha_data.flatten(), bins=80, color='c')
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.show()

# 使用你的DICOM文件路径
# dicom_path = r'E:\dataset\Clinic_data\2023021508\CBCTp5'
# plot_dicom_histogram(dicom_path)
# # CT_min_num,CT_max_num,len_num = dicom_read_max_min(dicom_path)
# # print('CT_min_num:',CT_min_num,'CT_max_num:',CT_max_num,'len_num:',len_num)
# 使用你的PNG文件路径
# png_path = r'E:\dataset\temp_png\112HM10395\CBCTpriorAVG'
# plot_png_histogram(png_path)

# 使用你的NPY文件路径
# npy_path = r'\\192.168.202.30\FtpWorkDir\SaveBibMip-SX\eva_data\PRIOR_test\Sparse\phase_4'
# plot_npy_histogram(npy_path)

# npy_path = r'E:\dataset\temp_npy\112HM10395\CBCTpriorAVG'
# plot_npy_histogram(npy_path)
#使用mha文件路径
# mha_path = r'E:\dataset\temp_dicom\100HM10395\CBCTp1.mha'
# plot_mha_histogram(mha_path)

# import numpy as np
# import matplotlib.pyplot as plt
# mha_file = r'E:\dataset\temp_dicom\100HM10395\CBCTp1.mha'
# image = sitk.ReadImage(mha_file)
# img_data = sitk.GetArrayFromImage(image).transpose((2, 1, 0))

# #计算img_data的灰度值范围
# max_CT_num = np.max(img_data)
# min_CT_num = np.min(img_data)
# print('max_CT_num:',max_CT_num,'min_CT_num:',min_CT_num)
# #计算img_data的灰度值最集中的区间
# img_data = img_data.flatten()
# img_data = img_data[img_data>0]
# img_data = img_data[img_data<2000]
# plt.hist(img_data, bins=80, color='c')
# plt.xlabel("Hounsfield Units (HU)")
# plt.ylabel("Frequency")
# plt.show()