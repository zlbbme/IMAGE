import odl
import numpy as np
import pydicom
import os
from odl.contrib import tomo

import matplotlib.pyplot as plt
#导入峰值信噪比模块
from skimage.metrics import peak_signal_noise_ratio as PSNR
#导入结构相似度模块
from skimage.metrics import structural_similarity as SSIM

# 读取DICOM文件并获取图像数据的函数
def read_dicom(path_to_dicom_file):
    ds = pydicom.dcmread(path_to_dicom_file, force=True)
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    instance_number = ds.InstanceNumber
    image_data = ds.pixel_array.astype(np.int16)  # 转换为浮点类型
    return image_data

#浮点数扩维并拼接
def read_gate(path_gate):
    #遍历path_gate文件夹下的所有文件的完整路径名
    
    # 读取DICOM文件并获取图像数据
    for root, dirs, files in os.walk(path_gate):
        for i,file in enumerate (files):
            file_name = os.path.join(root, file)
        
            image_data = np.expand_dims(read_dicom(file_name), axis=2)  #[512,512,1]
            #按[x,y,z]方向重构
            if i==0:
                gate_data = image_data
            else:
                gate_data = np.concatenate((gate_data,image_data),axis=2)

                #print(image_data.shape)
            
    return gate_data


if __name__ == '__main__':
    patient_file = r'E:\dataset\Clinic_data\2022090604'
    gate_path = patient_file + '\CTp0'
    #get gated image data
    gate_data = read_gate(gate_path) # [512,512,60]
    print('input_data_shape:',gate_data.shape)
    print('data range from %d to %d'%(gate_data.min(),gate_data.max()))
    # 定义重建空间和投影几何
    space = tomo.elekta_xvi_space(shape= gate_data.shape)
    detector_geometry = tomo.elekta_xvi_geometry(angles = np.linspace(0, 2 * np.pi, 60, endpoint=False),#num_angles=600,
                                                piercing_point=(512.0, 512.0),detector_shape=(1024, 1024))
    print('detector.shape:',detector_geometry.detector.shape)     # (1024, 1024)

    # Create ray transform
    ray_transform = odl.tomo.RayTransform(space, detector_geometry, use_cache=False)
    # Create artificial data
    projections = ray_transform(gate_data)     # (60, 1024, 1024), 60 projections ,float32

    # Get default FDK reconstruction operator
    recon_op = tomo.elekta_xvi_fbp(ray_transform)

    reconstruction = recon_op(projections).astype(np.int16)
    #转换成numpy数组
    reconstruction = reconstruction.asarray()
    reconstruction[reconstruction <0] = 0 
    #归一到0-最大值
    #reconstruction = (reconstruction - reconstruction.min())/(reconstruction.max() - reconstruction.min())*4000

    print(reconstruction.shape)  # (512, 512, 60) int16
    #打印最大值
    print('data range from %d to %d'%(reconstruction.min(),reconstruction.max()))
    #projections.show('projections')  #显示投影正弦图
    
    #通过512*512的灰度图片显示gate_date的三个方向的投影

    # plt.imshow(gate_data[:,:,24],cmap='gray')
    # plt.show()

    # plt.imshow(reconstruction[:,:,24],cmap='gray')
    # plt.show()
    #两幅图像并列显示
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(gate_data[:,:,24],cmap='gray')
    ax[1].imshow(reconstruction[:,:,24],cmap='gray')
    plt.show()

    hist, bins = np.histogram(gate_data[:,:,24], bins=50)
    #打印gate_data[:,:,24]的灰度直方图
    plt.hist(gate_data[:,:,24].flatten(), bins=bins)
    plt.show()
    plt.hist(reconstruction[:,:,24].flatten(), bins=bins)
    plt.show()
    
    # #两幅图像显示灰度直方图
    # fig, ax = plt.subplots(1, 2)
    # ax[0].hist(gate_data[:,:,24].flatten(), bins=50)
    # ax[0].set_title('Original')
    # ax[1].hist(reconstruction[:,:,24].asarray().flatten(), bins=50)
    # ax[1].set_title('Reconstruction')
    # plt.show()
    
    #计算均方误差
    mse = np.mean((gate_data[:,:,24] - reconstruction[:,:,24])**2)
    print(mse)
    #计算峰值信噪比
    psnr = PSNR(gate_data[:,:,24],reconstruction[:,:,24],data_range=gate_data[:,:,24].max() - gate_data[:,:,24].min())
    print(psnr)
    #计算结构相似度
    ssim = SSIM(gate_data[:,:,24],reconstruction[:,:,24],data_range=gate_data[:,:,24].max() - gate_data[:,:,24].min())
    print(ssim)
    