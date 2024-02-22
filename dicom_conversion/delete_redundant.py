import os

def delete_file(folder_path,start_num,end_num,file_type):
    start_num = int(str(start_num).zfill(3))   ; end_num = int(str(end_num).zfill(3))
    #获取folder_path最底层目录
    for root, dirs, files in os.walk(folder_path):
        #如果files为空，则跳过
        if not files:
            continue
        #只要files为.DCM后缀文件，则获取上一层文件名
        if files[0].endswith(file_type):
            input_folder = root 
            #获取folder_path最底层目录下的所有文件名
            file_names = os.listdir(input_folder)
            #遍历所有文件名
            for file_name in file_names:
                #print(os.path.split(file_name))
                #获取文件名的数字
                file_num = int(str(os.path.split(file_name)[-1].split('.')[0]).zfill(3))
                #判断文件名的数字是否在start_num和end_num之间，数字为3个字符，不足三个字符的前面补0
                print(file_num,start_num,end_num)
                #print(file_num)
                if file_num in range(start_num,end_num):
                    # Delete the file
                    
                    os.remove(os.path.join(input_folder, file_name))
                    print('Delete the file: ',file_name)
    print('Delete Done!')

def rename_file(file_path,start_num,file_type):
    #获取folder_path最底层目录
    
    for root, dirs, files in os.walk(file_path):
        #如果files为空，则跳过
        if not files:
            continue
        #只要files为file_type后缀文件，则获取上一层文件名
        if files[0].endswith(file_type):
            input_folder = root 
            #获取folder_path最底层目录下的所有文件名
            file_names = os.listdir(input_folder)
            file_names.sort(key=lambda x: int(x[:-4]))
            print(file_names)
            temp_num = start_num
            #遍历所有文件名
            for file_name in file_names:
                #获取文件名的数字
                file_num = int(os.path.split(file_name)[-1].split('.')[0])
                
                #从小到大排序，将文件名从0开始，以1递增，用file_type作为后缀保存
                #保存数字为三个字符，不足三个字符的前面补0
                new_file_name = str(temp_num) + file_type  #.zfill(3)

                #Rename the file with a new name starting with 1
                os.rename(os.path.join(input_folder, file_name), os.path.join(input_folder, new_file_name))
                print('Rename the file:',file_name,' to ',new_file_name)
                temp_num += 1
            

    print('Rename Done!')

def Clinic_collate(Patient_ID,end1,start2):
    "retain files in [end1,start2] and rename the files in the folder!"
    
    folder_png_path = os.path.join(r'E:\dataset\temp_png',Patient_ID)
    folder_npy_path = os.path.join(r'E:\dataset\temp_npy',Patient_ID)
    rename_file(folder_png_path,0,file_type = '.png');              rename_file(folder_npy_path,0,file_type = '.npy')

    delete_file(folder_png_path,0,end1-1,file_type = '.png');    delete_file(folder_png_path,start2,1000,file_type = '.png')

    delete_file(folder_npy_path,0,end1-1,file_type = '.npy');    delete_file(folder_npy_path,start2,1000,file_type = '.npy')

    rename_file(folder_png_path,0,file_type = '.png');rename_file(folder_npy_path,0,file_type = '.npy')

    print('Collate Done!\n Please check the folder:',folder_png_path,' and ',folder_npy_path,' for the results!')

#定义一个函数，文件名中的字母去除，只留下数字部分
def cut_english(folder_path,file_type):
    #获取folder_path最底层目录
    for root, dirs, files in os.walk(folder_path):
        #如果files为空，则跳过
        if not files:
            continue
        #只要files为.DCM后缀文件，则获取上一层文件名
        if files[0].endswith(file_type):
            input_folder = root 
            #获取folder_path最底层目录下的所有文件名
            file_names = os.listdir(input_folder)
            #遍历所有文件名
            for file_name in file_names:
                #将文件名中字母部分去除，只留下数字部分
                new_file_name = ''.join([i for i in file_name if i.isdigit()]) + file_type
                #重命名文件
                os.rename(os.path.join(input_folder, file_name), os.path.join(input_folder, new_file_name))
                print('Rename the file:',file_name,' to ',new_file_name)

def rename_folder(root_path):
    #获取root_path最底层目录
    for root, dirs, files in os.walk(root_path):
        #如果files为空，则跳过
        if not files:
            continue
        #只要files为.DCM后缀文件，则获取上一层文件名
        if files[0].endswith(".npy"):
            input_folder = root 
            #如果input_folder中包含'degraded'，则将'degraded'替换为'CBCT'
            if 'DegradePhase' in input_folder:
                new_folder = input_folder.replace('DegradePhase','CBCTp')
                os.rename(input_folder, new_folder)
                print('Rename the folder:',input_folder,' to ',new_folder)
            if 'GT_Phase' in input_folder:
                new_folder = input_folder.replace('GT_Phase','CTp')
                os.rename(input_folder, new_folder)
                print('Rename the folder:',input_folder,' to ',new_folder)
            if input_folder.split('\\')[-1]=='Prior':
                new_folder = input_folder.replace('Prior','CBCTpriorAVG')
                os.rename(input_folder, new_folder)
                print('Rename the folder:',input_folder,' to ',new_folder)
            

if __name__ == '__main__':
    # for i in range (10):
        
    #     folder_path = r'/Data/SaveBibMip-SX/eva_data/Clinic_data/test_clinic/CTp'+str(i) ;file_type = '.npy'
    #     rename_file(folder_path,1,file_type)
        #delete_file(folder_path,44,76,file_type)
    #     delete_file(folder_path,44,76,file_type)#从0开始,从44开始删除
    # #读取网络文件夹
    # folder_path = r'E:\dataset\temp_new\new_DCM' ;file_type = '.DCM'
    # rename_file(folder_path,0,file_type)
    # Patient_ID = '2023022304'
    # Clinic_collate(Patient_ID,19,62) #retain files in [19,62] and rename the files in the folder!

    #去除多余字母
    folder_path = r'\\192.168.202.30\FtpWorkDir\SaveBibMip-SX\Training_data\sim_data' ;file_type = '.npy'
    #cut_english(folder_path,file_type)
    #rename_file(folder_path,0,file_type)
    rename_folder(folder_path)