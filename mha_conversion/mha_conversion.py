import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pydicom

def convert_mha_to_png(mha_file, path_png):
    image = sitk.ReadImage(mha_file)
    img_data = sitk.GetArrayFromImage(image)
    height = img_data.shape[0]
    weight = img_data.shape[1]
    channel = img_data.shape[2]
    
    #判断是否存在文件夹如果不存在则创建为文件夹
    if not os.path.exists(path_png):
        os.makedirs(path_png)

    for i in range(channel):
        img = np.zeros((height,weight), dtype=np.uint8)
        img = img_data[:,:,i]
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        #镜面翻转
        img = cv2.flip(img, 1)
        #顺时针旋转180度
        img = cv2.rotate(img, cv2.ROTATE_180)
        #重构为512*512
        img = cv2.resize(img,(512,512),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(path_png+'/'+str(i+1)+'.png',img)
        print('from mha to png',i+1,'/',channel)

def convert_mha_to_npy(mha_file, npy_path):
    image = sitk.ReadImage(mha_file)
    img_data = sitk.GetArrayFromImage(image)  #[H,W,D]
    print(img_data.shape)
    height = img_data.shape[0]
    weight = img_data.shape[1]
    channel = img_data.shape[2]
    
    #判断是否存在文件夹如果不存在则创建为文件夹
    if not os.path.exists(npy_path):
        os.makedirs(npy_path)

    for i in range(channel):
        img = np.zeros((height,weight), dtype=np.float32)
        img = img_data[:,:,i]
        #顺时针旋转180度
        img = cv2.rotate(img, cv2.ROTATE_180)
        #重构为512*512
        img = cv2.resize(img,(512,512),interpolation=cv2.INTER_CUBIC)
        #保存为mha文件
        np.save(npy_path+'/'+str(i+1)+'.npy',img)
        print('from mha to npy',i+1,'/',channel)

def mha_to_direct(inpput_mha,output_mha):
    # 读取第一个.mha图像
    image = sitk.ReadImage(inpput_mha,outputPixelType=sitk.sitkFloat32)
    #变换mha的方向，从[H,D,W]变成[H,W,D]
    out_image = sitk.PermuteAxes(image, [0,2,1])
    #保存为mha文件
    sitk.WriteImage(out_image, output_mha)
    #打印维度变化
    print('from',image.GetSize(),'to',out_image.GetSize())

def mha_to_equal(input_mha,fixed_mha,output_mha):
    # 读取第一个.mha图像
    fixed_image = sitk.ReadImage(fixed_mha,outputPixelType=sitk.sitkFloat32)
    moving_image = sitk.ReadImage(input_mha,outputPixelType=sitk.sitkFloat32)

    # 创建配准对象
    registration_method = sitk.ImageRegistrationMethod()

    # 设置配准方法和参数
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=0.1, minStep=1e-4, numberOfIterations=1)#只迭代一次
    registration_method.SetInitialTransform(sitk.TranslationTransform(fixed_image.GetDimension()))

    # 执行配准
    transform = registration_method.Execute(fixed_image, moving_image)

    # 应用变换到移动图像
    registered_image = sitk.Resample(moving_image, fixed_image, transform, sitk.sitkLinear, 0.0)

    # 保存配准后的图像，确保和fixed_image具有相同的原点、方向和间距和维度
    sitk.WriteImage(registered_image, output_mha)
    #打印维度变化
    print('from',moving_image.GetSize(),'to',registered_image.GetSize())

if __name__ == '__main__':

    def mha_batch(mha_path):
        mha_list = os.listdir(mha_path)
        os.makedirs(os.path.join(mha_path,'direct'))
        output_path = os.path.join(mha_path,'direct')
        for mha in mha_list:
            #判断是否为文件
            if not os.path.isfile(os.path.join(mha_path,mha)):
                continue
            input_mha = os.path.join(mha_path,mha)
            output_mha = os.path.join(output_path,mha)
            mha_to_direct(input_mha,output_mha)

    mha_path = r'E:\dataset\temp_mha\P3'
    mha_batch(mha_path)