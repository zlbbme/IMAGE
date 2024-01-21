import os

def delete_file(folder_path,start_num,end_num,file_type):
    
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
                file_num = int(os.path.split(file_name)[-1].split('.')[0])
                #print(file_num)
                if file_num in range(start_num,end_num):
                    # Delete the file
                    print('Delete the file: ',file_name)
                    os.remove(os.path.join(input_folder, file_name))
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
            #遍历所有文件名
            for file_name in file_names:
                #获取文件名的数字
                file_num = int(os.path.split(file_name)[-1].split('.')[0])

                #从小到大排序，将文件名从1开始，以1递增，用file_type作为后缀保存
                new_file_name = str(start_num) + file_type

                # Rename the file with a new name starting with 1
                os.rename(os.path.join(input_folder, file_name), os.path.join(input_folder, new_file_name))
                start_num+=1

    print('Rename Done!')
if __name__ == '__main__':
    folder_path = '2021121308_png/CTAVG' 
    #delete_file(folder_path,1,14,'.png')
    rename_file(folder_path,1,'.png')