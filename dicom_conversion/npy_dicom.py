import numpy as np
import os
import pydicom
import nibabel as nib
import re

#dicom路径
dicom_file = r'E:\dataset\temp_delete\dcm\0.dcm'
#npy路径
npy_file = dicom_file.replace('dcm','npy')

#将读取的npy数组赋值给dicom文件
def npy2dicom(dicom_file, npy_file):
    #读取dicom文件
    ds = pydicom.dcmread(dicom_file, force=True)  
    npy = np.load(npy_file).astype(np.uint16)
    #将npy数组赋值给dicom文件
    ds.PixelData = npy.tobytes()
    dicom_new_file = dicom_file.replace('temp_delete','temp_new')
    if not os.path.exists(os.path.split(dicom_new_file)[0]):
        os.makedirs(os.path.split(dicom_new_file)[0])
    #保存dicom文件
    ds.save_as(dicom_new_file)

#write_dicom(dicom_file, npy_file)

def batch_npy2dicom(dicom_path):
    for files in os.listdir(dicom_path):
        dicom_file = os.path.join(dicom_path,files)
        npy_file = dicom_file.replace('dcm','npy')
        npy2dicom(dicom_file, npy_file)


def npy2nii(npy_path):
    file_nums = len(os.listdir(npy_path))
    volumn_data = np.zeros((512, 512, file_nums), dtype=np.int16)
    
    for npy_file in os.listdir(npy_path):
        #获取npy_file中的数字
        num = int(re.findall(r'\d+', npy_file)[0])
        
        npy_data = np.load(os.path.join(npy_path,npy_file)).astype(np.int16)
        #创建空的npy数组
        volumn_data [:,:,num]= npy_data
    
    #创建nii文件
    nii_file = os.path.join(npy_path,os.path.split(npy_path)[-1]+'.nii.gz')
    #将npy数组转换为nii文件
    img = nib.Nifti1Image(volumn_data, np.eye(4))
    #保存nii文件
    nib.save(img, nii_file)
    print('npy to nii is done!\n %s has write'%(nii_file))

#截断npy文件的灰度值到[0,2500]
def truncate_npy(npy_path):
    print('Truncate the npy file to [0,1200]!')
    for npy_file in os.listdir(npy_path):
        npy_data = np.load(os.path.join(npy_path,npy_file))
        npy_data[npy_data<20] = 0;   npy_data[npy_data>2500] = 2500
        #归一到[0,2500]
        npy_data = (npy_data/2500)*1200
        np.save(os.path.join(npy_path,npy_file),npy_data) #覆盖原文件
        print('Truncate the npy file:',npy_file)

if __name__ == '__main__':
    print('Let\'s npy to dicom!')
    for i in range (10):
        dicom_path = r'E:\dataset\temp_delete\dcm\CBCTp'+str(i)
        batch_npy2dicom(dicom_path)
    # dicom_path = r'E:\dataset\temp_delete\dcm'
    # batch_npy2dicom(dicom_path)
    # npy_path = r'E:\dataset\temp_npy\100HM10395\CBCTp0'
    # npy2nii(npy_path)
    
    # path = r'E:\dataset\temp_npy\test_clinic\Lastresult'
    # for phase_path in os.listdir(path):
    #     npy_path = os.path.join(path,phase_path)
    #     truncate_npy(npy_path)
        