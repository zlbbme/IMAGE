import os
from PIL import Image
import SimpleITK as sitk
import numpy as np
import pydicom
import matplotlib.pyplot as plt

def convert_dicom_to_png(dicom_folder, output_folder):
    #判断输出文件夹是否存在，不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    min_CT_num,max_CT_num,slice_num  = dicom_read_max_min(dicom_folder)
    #遍历dicom文件夹下的所有文件，如果是dicom文件则转换为png文件
    for root, dirs, files in os.walk(dicom_folder):
         
        for file in files:
            if file.endswith(".DCM"):
                dicom_path = os.path.join(root, file)
                print(dicom_path)
                
            # 读取 DICOM 文件
            ds = pydicom.dcmread(dicom_path, force=True)
            ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
            #获取dicom文件的instance number
            instance_number = ds.InstanceNumber
            # 将像素值的范围调整到 [0, 255]
            pixel_array = ds.pixel_array

            pixel_array = (pixel_array - min_CT_num) / (max_CT_num - min_CT_num) * 255
            #print(np.min(pixel_array), np.max(pixel_array))
            pixel_array = pixel_array.astype(np.uint8)
 
            # 创建 PIL Image 对象
            image = Image.fromarray(pixel_array)

            # 保存为 PNG 文件
            print(os.path.splitext(file)[0])
            png_filename = str(slice_num-instance_number) + '.png'
            png_filepath = os.path.join(output_folder, png_filename)
            image.save(png_filepath)

def convert_dicom_to_npy(dicom_folder, output_folder):
    #判断输出文件夹是否存在，不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    _,_,slice_num  = dicom_read_max_min(dicom_folder)
    #遍历dicom文件夹下的所有文件，如果是dicom文件则转换为png文件
    for root, dirs, files in os.walk(dicom_folder):

        for file in files:
            if file.endswith(".DCM"):
                dicom_path = os.path.join(root, file)
                print(dicom_path)
                
            # 读取 DICOM 文件
            ds = pydicom.dcmread(dicom_path, force=True)
            ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian  #‘FileMetaDataset’ object has no attribute ‘TransferSyntaxUID’ 错误
            instance_number = ds.InstanceNumber
            #将数据格式转换为numpy
            pixel_array = ds.pixel_array.astype(np.float32)
            #保存为npy文件
            np_filename = str(slice_num-instance_number) + '.npy'
            np_filepath = os.path.join(output_folder, np_filename)
            np.save(np_filepath, pixel_array)


def convert_dicom_to_nrrd(input_folder, output_nrrd):
    dicom_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(input_folder)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(dicom_names)
    dicom_series = reader.Execute()
    sitk.WriteImage(dicom_series, output_nrrd)

def convert_dicom_to_mha(input_folder, output_mha):
    dicom_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(input_folder)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(dicom_names)
    dicom_series = reader.Execute()
    sitk.WriteImage(dicom_series, output_mha)

def convert_dicom_to_nii(input_folder, output_nii):
    dicom_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(input_folder)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(dicom_names)
    dicom_series = reader.Execute()
    sitk.WriteImage(dicom_series, output_nii)

def dicom_read_max_min(dicom_path):
    max_list = []; min_list = [] ;len_dicom = 0
    #遍历文件夹下的所有文件，获取后缀为.DCM的文件
    for root, dirs, files in os.walk(dicom_path):
        for file in files:
            if file.endswith(".DCM"):
                len_dicom += 1
                dicom_path = os.path.join(root, file)
                print(dicom_path)
            # 读取 DICOM 文件
            ds = pydicom.dcmread(dicom_path, force=True)
            ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
            #获取dicom文件的instance number
            instance_number = ds.InstanceNumber
            pixel_array = ds.pixel_array
            max_list.append(np.max(pixel_array))
            min_list.append(np.min(pixel_array))
    max_CT_num = np.max(max_list)
    min_CT_num = np.min(min_list)
    return min_CT_num, max_CT_num, len_dicom

if __name__ == "__main__":
    # Usage example
    dicom_folder = "./2021121308/CTAVG"
    output_folder = 'png'    
    #convert_dicom_to_png(dicom_folder, output_folder)
    #convert_dicom_to_npy(dicom_folder, output_folder)
    #dicom_series_to_nrrd(dicom_folder, output_folder)
    #读取npy文件并显示
    data = np.load("./npy/39.npy")
    #可视化显示
    plt.imshow(data, cmap='gray')
    plt.show()
    #convert_dicom_to_mha(dicom_folder, output_folder)
    #convert_dicom_to_nii(dicom_folder, output_folder)
    #max_CT_num, min_CT_num =dicom_read_max_min(dicom_folder)
    #print(max_CT_num, min_CT_num)