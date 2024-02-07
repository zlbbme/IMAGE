import batch_conversion
import os
import mha_conversion
import shutil
def TCIA_npy(patient_path,out_path):
    #遍历dicom_folder下的所有文件
    for root, dirs, files in os.walk(patient_path):
        #如果files为空，则跳过
        if not files:
            continue
        #只要files为.DCM后缀文件，则获取上一层文件名
        if files[0].endswith(".DCM"):
            dicom_folder = root ; 
        out_mha_path = patient_path.replace('temp_dicom','temp_mha')
        batch_conversion.batch_conversion(dicom_folder, out_mha_path, 'mha')
    print('mha conversion is done!')
    for root, dirs, files in os.walk(out_mha_path):
        #如果files为空，则跳过
        if not files:
            continue
        #只要files为.mha后缀文件，则获取文件名
        if files:
            for file in files:
                mha_file = os.path.join(root,file)
                
                out_put_path = out_path + '\\' + os.path.splitext(file)[0]
                
                if 'CBCT' in mha_file :
                    print(file,'converting to npy')
                    mha_conversion.convert_mha_to_npy(mha_file, out_put_path)
                    continue
                else:
                    CT_mha_file = mha_file
                    CBCT_mha_file = mha_file.replace('CT','CBCT')
                    
                    print(file,'converting to npy')
                    mha_conversion.mha_to_equal(CT_mha_file,CBCT_mha_file,CT_mha_file)
                    mha_conversion.convert_mha_to_npy(CT_mha_file, out_put_path)

    print('npy conversion is done!')


def TCIA_png(patient_path,out_path):
    #遍历patient_path下的所有文件
    for root, dirs, files in os.walk(patient_path):
        #如果files为空，则跳过
        if not files:
            continue
        #只要files为.DCM后缀文件，则获取上一层文件名
        if files[0].endswith(".DCM"):
            dicom_folder = root ; 
        out_mha_path = patient_path.replace('temp_dicom','temp_mha')
        batch_conversion.batch_conversion(dicom_folder, out_mha_path, 'mha')
    print('mha conversion is done!')
    for root, dirs, files in os.walk(out_mha_path):
        #如果files为空，则跳过
        if not files:
            continue

        if files:
            for file in files:
                mha_file = os.path.join(root,file)
                
                out_put_path = out_path + '\\' + os.path.splitext(file)[0]
                
                if 'CBCT' in mha_file :
                    print(file,'converting to png')
                    mha_conversion.convert_mha_to_png(mha_file, out_put_path)
                    continue
                else:
                    CT_mha_file = mha_file
                    CBCT_mha_file = mha_file.replace('CT','CBCT')
                    
                    print(file,'converting to png')
                    mha_conversion.mha_to_equal(CT_mha_file,CBCT_mha_file,CT_mha_file)
                    mha_conversion.convert_mha_to_png(CT_mha_file, out_put_path)
    
    print('png conversion is done!')

def Clinic_png(patient_path):
    #遍历patient_path下的所有文件
    for root, dirs, files in os.walk(patient_path):
        #如果files为空，则跳过
        if not files:
            continue
        #只要files为.DCM后缀文件，则获取上一层文件名
        if files[0].endswith(".DCM"):
            dicom_folder = root ; 
            
        out_png_path = patient_path.replace('temp_dicom','temp_png')
        print(dicom_folder,out_png_path)
        batch_conversion.batch_conversion(dicom_folder, out_png_path, 'png')

def Clinic_npy(patient_path):
    #遍历patient_path下的所有文件
    for root, dirs, files in os.walk(patient_path):
        #如果files为空，则跳过
        if not files:
            continue
        #只要files为.DCM后缀文件，则获取上一层文件名
        if files[0].endswith(".DCM"):
            dicom_folder = root ; 
            
        out_png_path = patient_path.replace('temp_dicom','temp_npy')
        print(dicom_folder,out_png_path)
        batch_conversion.batch_conversion(dicom_folder, out_png_path, 'npy')


if __name__ == '__main__':
    print('Let\'s start!')
    # for i in range (20):
        
    #     TCIA_PATH = r'E:\dataset\temp_dicom'
    #     patient_path = TCIA_PATH+'\\1'+'%02dHM10395'%(i) 
    #     #print(patient_path)
    #     out_path = patient_path.replace('temp_dicom','temp_png')
    #     TCIA_png(patient_path,out_path)
    #     out_path = patient_path.replace('temp_dicom','temp_npy')
    #     TCIA_npy(patient_path,out_path)
    #     temp_mha_path = patient_path.replace('temp_dicom','temp_mha')
    #     shutil.rmtree(temp_mha_path)
    #     #完成进度条
    #     print('The',i,'th patient is done!')
    # patient_path = r'E:\dataset\Clinic_data\2021121308'
    # Clinic_npy(patient_path)
    Clinic_path = r'E:\dataset\temp_dicom'
    for patient in os.listdir(Clinic_path):
        patient_path = os.path.join(Clinic_path,patient)
        
        Clinic_png(patient_path)
        
        Clinic_npy(patient_path)
        #完成进度条
        print(patient,'is done!')