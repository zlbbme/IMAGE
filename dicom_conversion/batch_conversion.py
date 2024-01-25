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
    #batch_conversion(r'E:\dataset\Clinic_data\2021121308', r'E:\dataset\Clinic_data\2021121308_png', 'png')
    input_dicom_path = r'E:\dataset\temp_dicom\100HM10395'
    output_path = r'E:\dataset\temp_dicom\100HM10395\png'
    batch_conversion(input_dicom_path, output_path, 'png')