import numpy as np
import matplotlib.pyplot as plt

import pydicom
from dicom_conversion import *

# path =r'E:\data patient\2022032401\1'
# #遍历path下的所有文件
# for root, dirs, files in os.walk(path):
#     for file in files:
#         if file.endswith(".DCM"):
#             data = os.path.join(root, file)
#             #读取data
#             ds = pydicom.dcmread(data, force=True)
#             #获得instance number
#             instance_number = ds.InstanceNumber
#             #如果instance number为10，则打印文件路径
#             if instance_number == 68:
#                 a,b,c = dicom_read_max_min(data)
#                 print(a,b,c)
# #                 print(data)
            
# npy_data = np.load(r'E:\dataset\temp_dicom\100HM10395\npy\CTp0\10.npy')
# # print(npy_data.min(),npy_data.max())
# # # 使用matplotlib进行可视化
# plt.imshow(npy_data,'gray')
# plt.show()


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
        if instance_number == 50:
            print(dicom_file_path)
            #获取dicom文件的像素值
            pixel_array = ds.pixel_array
            #绘制直方图
            plt.hist(pixel_array.flatten(), bins=80, color='c')
            plt.xlabel("Hounsfield Units (HU)")
            plt.ylabel("Frequency")
            plt.show()
            break
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
        break

# # # 使用你的DICOM文件路径
# dicom_path = r'E:\dataset\temp_dicom\100HM10395\CTp0'
# plot_dicom_histogram(dicom_path)

# # 使用你的PNG文件路径
png_path = r'E:\dataset\Clinic_data\2021121308\png\CBCTp40'
plot_png_histogram(png_path)

# # 使用你的NPY文件路径
npy_path = r'E:\dataset\Dataset_For_PRIOR\test\Label\phase_1'
plot_npy_histogram(npy_path)

#取出0-141中的中间50个数
# a = np.arange(0,142)
# print(a)
# print(a[46:96])