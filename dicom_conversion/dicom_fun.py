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
#                 print(data)
            
# npy_data = np.load(r'E:\data patient\2022032401_npy\1\10.npy')
# print(npy_data.min(),npy_data.max())
# # 使用matplotlib进行可视化
# plt.imshow(npy_data,'gray')
# plt.show()


dicom_path = r'E:\dataset\Clinic_data\2021121308\CBCTAVG'

dicom_files = os.listdir(dicom_path)

for dicom_dile in dicom_files:
    dicom_file_path = os.path.join(dicom_path,dicom_dile)
    ds =pydicom.dcmread(dicom_file_path,force=True)
    print(ds.InstanceNumber)
    if ds.InstanceNumber == 75:
        print('ok')
        break

#CBCT的 instance number 是从0-74
#CT的 instance number 是从1-75