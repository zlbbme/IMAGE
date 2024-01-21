import numpy as np
import os
#导入读取图片的包
from PIL import Image

#读取Degraded文件夹下的图片
def read_image(path):
    #获取Degraded文件夹下的所有文件名
    filelist = os.listdir(path)
    #创建一个空列表，用于存储图片
    image_list = []
    #创建一个空的三维数组，用于存储图片
    img_all = np.zeros((512, 512, 1))                    
    #遍历所有文件名
    for file in filelist:
        #将图片的路径和文件名拼接起来
        image_path = os.path.join(path, file)
        #打开图片
        img = Image.open(image_path)
        #将图片转换为数组
        img = np.array(img).reshape(512, 512, 1)
        #print(img.shape)
        #将图片数组添加到列表中
        img_all = np.concatenate((img_all, img), axis=2)
        #print(img_all.shape)
    #返回图片列表
    return img_all

#img = read_image('./DegradePhase1')
#创建空的四维数组，用于存储图片
#image_phase = np.zeros((1, 512, 512, 88))
for i in range (10):
    path = './DegradePhase'+str(i+1)
    #image_phase=np.concatenate(image_phase,read_image(path).reshape(1,512, 512,88),axis=0)
    image_phase =np.expand_dims(read_image(path),axis=0)
    if i ==0:
        image_sum = image_phase
    else:
        image_sum = np.concatenate((image_sum, image_phase), axis=0)
    print(image_phase.shape)
    print(image_sum.shape)

image_avg = np.mean(image_sum, axis=0)
print(image_avg.shape)

for j in range(88):

    # #将numpy数组转换为图片
    png_avg = image_avg[:,:,j]
    #将png_avg转为位深度为8的灰度图
    png_avg = Image.fromarray(np.uint8(png_avg))#.convert('L')
    # #保存图片
    png_avg.save('./image_avg'+str(j+1)+'.png')
   
    