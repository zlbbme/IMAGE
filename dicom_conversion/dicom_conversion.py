import os
from PIL import Image
import SimpleITK as sitk
import numpy as np
import pydicom

#定义函数，使用SimpleITK将dicom序列转换为png文件
def dicom_series_to_png(dicom_folder, output_folder):
    #判断输出文件夹是否存在，不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    #读取dicom序列
    dicom_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dicom_folder)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(dicom_names)
    dicom_series = reader.Execute()
    #获取dicom序列的像素值
    dicom_array = sitk.GetArrayFromImage(dicom_series)
    #获取dicom序列的维度
    dicom_array_shape = dicom_array.shape
    print(dicom_array_shape)
    #获取图像的最大值和最小值
    max_CT_num = np.max(dicom_array); min_CT_num = np.min(dicom_array)
    #遍历dicom序列的每一张图像
    for i in range(dicom_array_shape[0]):
        
        #创建PIL Image对象
        image = Image.fromarray(dicom_array[i])
        #归一化图像
        image = (image-min_CT_num) / (max_CT_num - min_CT_num) * 255
        #转换图像数据类型
        image = Image.fromarray(image.astype('uint8'))
        #保存为png文件
        png_filename = str(i) + '.png'
        png_filepath = os.path.join(output_folder, png_filename)
        image.save(png_filepath)

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
                #print(dicom_path)
                
            # 读取 DICOM 文件
            ds = pydicom.dcmread(dicom_path, force=True)
            ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
            #获取dicom文件的instance number
            instance_number = ds.InstanceNumber
            # 将像素值的范围调整到 [0, 255]
            pixel_array = ds.pixel_array
            #将pixel_array中大于max_CT_num的值设置为max_CT_num，小于min_CT_num的值设置为min_CT_num
            pixel_array[pixel_array > max_CT_num] = max_CT_num; pixel_array[pixel_array < min_CT_num] = min_CT_num
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
                #print(dicom_path)
                
            # 读取 DICOM 文件
            ds = pydicom.dcmread(dicom_path, force=True)
            ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian  #‘FileMetaDataset’ object has no attribute ‘TransferSyntaxUID’ 错误
            instance_number = ds.InstanceNumber
            #将数据格式转换为numpy
            pixel_array = ds.pixel_array.astype(np.int16)
            pixel_array[pixel_array > max_CT_num] = max_CT_num; pixel_array[pixel_array < min_CT_num] = min_CT_num
            pixel_array = pixel_array - min_CT_num    #获取为相对电子密度，无须归一化
            #pixel_array = (pixel_array - min_CT_num) / (max_CT_num - min_CT_num) * 3000   #归一到0-4000
            #保存为npy文件
            print(os.path.splitext(file)[0])
            np_filename = str(slice_num-instance_number) + '.npy'
            np_filepath = os.path.join(output_folder, np_filename)
            np.save(np_filepath, pixel_array)


def convert_dicom_to_nrrd(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    dicom_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(input_folder)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(dicom_names)
    dicom_series = reader.Execute()
    output_nrrd = os.path.join(output_folder,os.path.split(input_folder)[-1] + '.nrrd')
    sitk.WriteImage(dicom_series, output_nrrd)

def convert_dicom_to_mha(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    dicom_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(input_folder)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(dicom_names)
    dicom_series = reader.Execute()
    #输出文件名为输入文件夹的文件夹名
    output_mha = os.path.join(output_folder,os.path.split(input_folder)[-1] + '.mha')
    sitk.WriteImage(dicom_series, output_mha)

def convert_dicom_to_nii(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    dicom_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(input_folder)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(dicom_names)
    dicom_series = reader.Execute()
    #输出文件名为输入文件夹的文件夹名
    output_nii = os.path.join(output_folder,os.path.split(input_folder)[-1] + '.nii')
    sitk.WriteImage(dicom_series, output_nii)

def dicom_read_max_min(dicom_path):
    max_list = []; min_list = [] ;len_dicom = 0; pixel_list = []
    #如果dicom_path是文件
    if os.path.isfile(dicom_path):
        # 读取 DICOM 文件
        ds = pydicom.dcmread(dicom_path, force=True)
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        #获取dicom文件的instance number
        instance_number = ds.InstanceNumber
        pixel_array = ds.pixel_array
        pixel_list.append(pixel_array)
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
                    pixel_list.append(pixel_array)

                    max_list.append(np.max(pixel_array))
                    min_list.append(np.min(pixel_array))
    if np.min(max_list) > 500:
        print('max_CT_num:>500')
        max_CT_num = np.min(max_list) ; min_CT_num = np.max(min_list)   #获取最大的CT值为最大值列表的最小值，最小的CT值为最小值列表的最大值，压缩可用灰度区间
    else:
        max_CT_num = np.max(max_list) ; min_CT_num = np.min(min_list)
    #max_CT_num = np.percentile(max_list, 80) ; min_CT_num = np.percentile(min_list, 20)
    #print(np.percentile(max_list, 0), np.percentile(min_list, 100))
    print('max_CT_num:', max_CT_num, 'min_CT_num:', min_CT_num, 'len_dicom:', len_dicom)
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
                print(dicom_file_path)
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
    dicom_folder = r'E:\dataset\temp_dicom\100HM10395\CTp1'
    # output_folder = 'npy'    
    #convert_dicom_to_png(dicom_folder, output_folder)
    #convert_dicom_to_npy(dicom_folder, output_folder)
    #dicom_series_to_nrrd(dicom_folder, output_folder)
    #读取npy文件并显示
    # data = np.load('./npy/45.npy')
    # #可视化显示
    # plt.imshow(data, cmap='gray')
    # plt.show()
    # output_mha  = r'E:\dataset\temp_dicom\100HM10395\CBCTAVG\mha'
    # convert_dicom_to_mha(dicom_folder, output_mha)
    #convert_dicom_to_nii(dicom_folder, output_folder)
    # min_CT_num, max_CT_num, len_dicom =dicom_read_max_min(dicom_folder)
    # print(max_CT_num, min_CT_num)
    # normalize_dicom_intensity(dicom_folder, 0, 4000)
    # min_CT_num, max_CT_num, len_dicom =dicom_read_max_min(dicom_folder)
    # print(max_CT_num, min_CT_num)
    output_folder = r'E:\dataset\temp_dicom\100HM10395\CTp1_dcm_png'
    convert_dicom_to_png(dicom_folder, output_folder)