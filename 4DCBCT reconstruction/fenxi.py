#查看两幅图片的直方图
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
#导入读取图片的包
from PIL import Image

path = './Prior_ArtifactFree/Prior1.png'
#读取图片
img = cv2.imread(path,0)
path1 = './image_avg2.png'
#读取图片
img1 = cv2.imread(path1,0)

#对比显示两张图片的直方图
plt.hist(img.ravel(),256,[0,256],color='r')
plt.hist(img1.ravel(),256,[0,256],color='b')
#显示直方图
plt.show()
