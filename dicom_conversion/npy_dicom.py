import numpy as np
import os
import pydicom
 
#dicom路径
dicom_file = r'E:\dataset\temp_delete\dcm\0.dcm'
#npy路径
npy_file = dicom_file.replace('dcm','npy')

#print(dicom_file,npy_file)


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

if __name__ == '__main__':
    print('Let\'s npy to dicom!')
    dicom_path = r'E:\dataset\temp_delete\dcm'
    batch_npy2dicom(dicom_path)
    