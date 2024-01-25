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
    min_CT_num,max_CT_num,slice_num  = dicom_read_max_min(dicom_folder)
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
            pixel_array = ds.pixel_array.astype(np.int16)

            pixel_array = (pixel_array - min_CT_num) / (max_CT_num - min_CT_num) * 4000   #归一到0-4000
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
    #如果dicom_path是文件
    if os.path.isfile(dicom_path):
        # 读取 DICOM 文件
        ds = pydicom.dcmread(dicom_path, force=True)
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        #获取dicom文件的instance number
        instance_number = ds.InstanceNumber
        pixel_array = ds.pixel_array
        max_list.append(np.max(pixel_array))
        min_list.append(np.min(pixel_array))
        len_dicom = 1

    if os.path.isdir(dicom_path):
    #遍历文件夹下的所有文件，获取后缀为.DCM的文件
        for root, dirs, files in os.walk(dicom_path):
            for file in files:
                if file.endswith(".DCM"):
                    len_dicom += 1
                    dicom_path = os.path.join(root, file)
                    #print(dicom_path)
                # 读取 DICOM 文件
                ds = pydicom.dcmread(dicom_path, force=True)
                ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
                #获取dicom文件的instance number
                instance_number = ds.InstanceNumber
                pixel_array = ds.pixel_array
                max_list.append(np.max(pixel_array))
                min_list.append(np.min(pixel_array))
                
    max_CT_num = np.max(max_list) ; min_CT_num = np.min(min_list)
    return min_CT_num, max_CT_num, len_dicom

def normalize_image_intensity(image, min_val, max_val):
    image = image.astype(np.float32)
    min_image = np.min(image)
    max_image = np.max(image)
    normalized_image = (image - min_image) / (max_image - min_image)  # 归一化到0-1
    normalized_image = normalized_image * (max_val - min_val) + min_val  # 缩放到min_val-max_val
    return normalized_image.astype(np.int16)

def normalize_dicom_intensity(dicom_path, min_val, max_val):
    for dicom_files in os.listdir(dicom_path):
        for dicom_file in dicom_files:
            if dicom_file.endswith(".DCM"):
                dicom_file_path = os.path.join(dicom_path, dicom_files)
                print(dicom_path)
    # 读取 DICOM 文件
                ds = pydicom.dcmread(dicom_file_path, force=True)
                ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
                #获取dicom文件的instance number
                instance_number = ds.InstanceNumber
                pixel_array = ds.pixel_array
                # 归一化图像
                normalized_pixel_array = normalize_image_intensity(pixel_array, 0, 4000)
                # 更新 DICOM 文件的像素数据
                ds.PixelData = normalized_pixel_array.tobytes()
                ds.Rows, ds.Columns = normalized_pixel_array.shape
                ds.PixelRepresentation = 1
                ds.BitsStored = 16
                ds.BitsAllocated = 16
                ds.HighBit = 15
                ds.SamplesPerPixel = 1
                ds.PhotometricInterpretation = "MONOCHROME2"
                ds.PixelData = normalized_pixel_array.tobytes()
                # 保存修改后的 DICOM 文件
                ds.save_as(dicom_file_path)
    

if __name__ == "__main__":
    # Usage example
    dicom_folder = r'E:\dataset\temp_dicom\100HM10395\CBCTp0'
    output_folder = 'npy'    
    #convert_dicom_to_png(dicom_folder, output_folder)
    #convert_dicom_to_npy(dicom_folder, output_folder)
    #dicom_series_to_nrrd(dicom_folder, output_folder)
    #读取npy文件并显示
    data = np.load('./npy/45.npy')
    #可视化显示
    plt.imshow(data, cmap='gray')
    plt.show()
    #convert_dicom_to_mha(dicom_folder, output_folder)
    #convert_dicom_to_nii(dicom_folder, output_folder)
    # min_CT_num, max_CT_num, len_dicom =dicom_read_max_min(dicom_folder)
    # print(max_CT_num, min_CT_num)
    # normalize_dicom_intensity(dicom_folder, 0, 4000)
    # min_CT_num, max_CT_num, len_dicom =dicom_read_max_min(dicom_folder)
    # print(max_CT_num, min_CT_num)