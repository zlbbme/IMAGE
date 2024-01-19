from dicom_conversion import *
import os
def batch_conversion(dicom_folder, output_folder, output_format):
    #获取dicom_folder下的所有文件
    files = os.listdir(dicom_folder)
    print(files)
    
    for file in files:
        dicom_files_name = os.path.join(dicom_folder, file)
        output_files_name = os.path.join(output_folder, file)
        print(dicom_files_name, output_files_name)
        
        if not os.path.exists(output_files_name):
            os.makedirs(output_files_name)
        if output_format == 'png':
            min_CT_num,  max_CT_num= dicom_read_max_min(dicom_files_name)
            convert_dicom_to_png(dicom_files_name, output_files_name, min_CT_num,max_CT_num)
        elif output_format == 'npy':
            convert_dicom_to_npy(dicom_files_name, output_files_name)
        elif output_format == 'mha':
            convert_dicom_to_mha(dicom_files_name, output_files_name)
        elif output_format == 'nrrd':
            convert_dicom_to_nrrd(dicom_files_name, output_files_name)
        elif output_format == 'nii':
            convert_dicom_to_nii(dicom_files_name, output_files_name)
        else:
            print('Don\'t support this format!  Please input png, npy, mha, nrrd or nii.')
            break


if __name__ == '__main__':
    print('Let\'s start!')
    batch_conversion('2021121308', '2021121308_copy', 'png')