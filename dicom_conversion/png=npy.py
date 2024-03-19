import os
import numpy as np
from PIL import Image


def convert_png_to_npy(image_path):
    #读取png文件
    image = Image.open(image_path)
    #将图像转换为numpy数组
    image_array = np.array(image).astype(np.float16)
    #归一化到0-2500
    image_array = image_array / 255 * 2500
    npy_path = image_path.replace('png', 'npy')
    #保存numpy数组为npy文件
    np.save(npy_path, image_array)


def batch_png2npy(input_path):

    
    # 遍历输入路径下的所有文件
    for root,dir,files in os.walk(input_path):
        for file in files:
            if file.endswith('.png'):
                # 读取图像
                image_path = os.path.join(root, file)
                output_path = root.replace('png', 'npy')
                print(output_path)
                if not os.path.exists(output_path):
                    os.makedirs(output_path, exist_ok=True)
                convert_png_to_npy(image_path)

def convert_npy_to_png(npy_path):
    #读取npy文件
    npy_file = np.load(npy_path)
    print(npy_file.min(),npy_file.max())
    npy_file[npy_file<30] = 0 ; npy_file[npy_file>npy_file.max()*0.8] = npy_file.max()*0.8
    print(npy_file.min(),npy_file.max())
    npy_file = (npy_file -npy_file.min()) / (npy_file.max()-npy_file.min()) * 255
    #将numpy数组转换为图像
    image = Image.fromarray(npy_file.astype(np.uint8))

    #保存图像
    #image.save(npy_path.replace('npy', 'png'))


def batch_npy2png(input_path):
    "npy to png"
    # 遍历输入路径下的所有文件
    for root,dir,files in os.walk(input_path):
        for file in files:
            if file.endswith('.npy'):
                # 读取图像
                npy_path = os.path.join(root, file)
                output_path = root.replace('npy', 'png')
                print(output_path)
                if not os.path.exists(output_path):
                    os.makedirs(output_path, exist_ok=True)
                convert_npy_to_png(npy_path)

if __name__ == '__main__':
    print('This function can convert npy=png')
    input_path = r'E:\dataset\sim_2021121308\CycNnet_result\Phase0'
    batch_npy2png(input_path)