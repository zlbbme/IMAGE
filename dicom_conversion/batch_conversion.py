from dicom_conversion import *
import os
def batch_conversion(input_folder, output_folder, output_format):

    #遍历dicom_folder下的所有文件
    for root, dirs, files in os.walk(input_folder):
        #如果files为空，则跳过
        if not files:
            continue
        #只要files为.DCM后缀文件，则获取上一层文件名
        if files[0].endswith(".DCM"):
            dicom_folder = root ; 
        #print(dicom_folder)
        #print(os.path.split(dicom_folder))
        output_dicom_folder = os.path.join(output_folder,os.path.split(dicom_folder)[-1])
        print(output_dicom_folder)
        if output_format == 'png':
            convert_dicom_to_png(dicom_folder, output_dicom_folder)
        elif output_format == 'npy':
            convert_dicom_to_npy(dicom_folder, output_dicom_folder)
        elif output_format == 'mha':
            convert_dicom_to_mha(dicom_folder, output_dicom_folder)
        elif output_format == 'nrrd':
            convert_dicom_to_nrrd(dicom_folder, output_dicom_folder)
        elif output_format == 'nii':
            convert_dicom_to_nii(dicom_folder, output_dicom_folder)
        else:
            print('Don\'t support this format!  Please input png, npy, mha, nrrd or nii.')
            break
            

if __name__ == '__main__':
    print('Let\'s start!')
    batch_conversion(r'E:\data patient\2022032401', r'E:\data patient\2022032401_copy', 'png')
    # dicom_path = r'E:\data patient\2022032401\1'
    # files = os.listdir(dicom_path)
    # instance = []
    # for file in files:
    #     if file.endswith(".DCM"):
    #         dicom_file = os.path.join(dicom_path, file)
    #         print(dicom_file)
    #         # 读取 DICOM 文件
    #         ds = pydicom.dcmread(dicom_file, force=True)
    #         ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    #         #获取dicom文件的instance number
    #         instance_number = ds.InstanceNumber
    #         instance = np.append(instance_number,instance)
    #         #顺序排列
    #         instance = np.sort(instance)

    # print(instance, len(instance))