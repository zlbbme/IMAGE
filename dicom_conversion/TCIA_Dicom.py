import batch_conversion
import os
def TCIA_npy(patient_path,out_npy_path):
    #遍历dicom_folder下的所有文件
    for root, dirs, files in os.walk(patient_path):
        #如果files为空，则跳过
        if not files:
            continue
        #只要files为.DCM后缀文件，则获取上一层文件名
        if files[0].endswith(".DCM"):
            dicom_folder = root ; 
        print(dicom_folder)


if __name__ == '__main__':
    print('Let\'s start!')
    patient_path = r'E:\dataset\temp_dicom\100HM10395'
    out_npy_path = r'E:\dataset\temp_npy\100HM10395'
    TCIA_npy(patient_path,out_npy_path)