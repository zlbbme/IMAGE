import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pydicom

def mha_read_max_min(mha_path):
    max_list = []; min_list = [] ;len_dicom = 0
    #如果mha_path是文件
    if os.path.isfile(mha_path):
        # 读取mha文件
        image = sitk.ReadImage(mha_path)
        img_data = sitk.GetArrayFromImage(image).transpose((2, 1, 0))
        max_list.append(np.max(img_data));min_list.append(np.min(img_data))
        max_CT_num = np.max(max_list)*0.8 ;min_CT_num = np.min(min_list)*0.8
        len_dicom = 1

    if os.path.isdir(mha_path):
    #遍历文件夹下的所有文件，获取后缀为.DCM的文件
        for root, dirs, files in os.walk(mha_path):
            for file in files:
                if file.endswith(".mha"):
                    len_dicom += 1
                    mha_path = os.path.join(root, file)
                    print(mha_path)
                # 读取 DICOM 文件
                    image = sitk.ReadImage(mha_path)
                    img_data = sitk.GetArrayFromImage(image).transpose((2, 1, 0))
                    max_list.append(np.max(img_data))
                    min_list.append(np.min(img_data))
                
    # if np.min(max_list) > 500:
    #     print('max CT value >500')
    #     max_CT_num = np.min(max_list) ; min_CT_num = np.max(min_list)   #获取最大的CT值为最大值列表的最小值，最小的CT值为最小值列表的最大值，压缩可用灰度区间
    # else:
    #max_CT_num = np.max(max_list) ; min_CT_num = np.min(min_list)
    #max_CT_num = np.percentile(max_list, 0.5) ; min_CT_num = np.percentile(min_list, 0.1)   #获取最大值和最小值的百分位数
    max_CT_num = 908 ; min_CT_num = -2096  #获取最大值和最小值的百分位数
    print('max_CT_num:',max_CT_num,'min_CT_num:',min_CT_num,'len_dicom:',len_dicom)
    return min_CT_num, max_CT_num, len_dicom

def mha_normalization(mha_file):
    # 读取图像
    image = sitk.ReadImage(mha_file)
    print(image.GetSize())
    # 创建一个重标定强度的过滤器
    rescale_filter = sitk.RescaleIntensityImageFilter()
    # 设置输出强度范围
    rescale_filter.SetOutputMinimum(0)
    rescale_filter.SetOutputMaximum(4000)
    # 应用过滤器
    rescaled_image = rescale_filter.Execute(image)
    # 保存结果,覆盖原文件
    sitk.WriteImage(rescaled_image, mha_file)

def mha_resample(mha_file):
    # 读取图像
    image = sitk.ReadImage(mha_file)

    # 获取原始图像的尺寸和间距
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()

    # 计算新的间距
    new_spacing = [os*sz/512 for os, sz in zip(original_spacing, original_size[:2])]
    # 创建一个重采样过滤器
    resampler = sitk.ResampleImageFilter()

    # 设置重采样的参数
    resampler.SetSize((512, 512, original_size[2]))
    resampler.SetOutputSpacing(new_spacing + [original_spacing[2]])
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)

    # 应用重采样过滤器
    resampled_image = resampler.Execute(image)

    # 保存结果
    sitk.WriteImage(resampled_image, mha_file)

def convert_mha_to_png(mha_file, path_png):
    #如果文件名为CBCT的mha文件，则最大值为3000，TCIA的数据集适用
    if 'CBCT' in mha_file:
        max_CT_num = 1000; min_CT_num = -2000
    else:
        max_CT_num = 500; min_CT_num = -1000
    # min_CT_num, max_CT_num, len_dicom = mha_read_max_min(mha_file)  #获取最大值
    # min_CT_num = float(min_CT_num);max_CT_num = float(max_CT_num)
    print(min_CT_num,max_CT_num)
    #读取mha文件
    image = sitk.ReadImage(mha_file)
    #将image中大于max_CT_num的值和小于min_CT_num的值赋值为min_CT_num
    #image = sitk.Threshold(image, min_CT_num, max_CT_num, min_CT_num)
    #重构方向
    #image = sitk.PermuteAxes(image, [0,2,1])   #变换mha的方向，从[H,D,W]变成[H,W,D]
    print('orgin mha shape',image.GetSize())
    img_data = sitk.GetArrayFromImage(image).transpose((2, 1, 0))  #转换成numpy，重构方向
    #归一从0到最大值
    img_data = img_data - min_CT_num
    #截取0到(max_CT_num-min_CT_num)
    if 'CBCT' in mha_file:
        img_data[img_data<0] = 0;  img_data[img_data>(max_CT_num-min_CT_num+100)] = 0
    else:
        img_data[img_data<0] = 0; 
    img_data[img_data>(max_CT_num-min_CT_num)] = max_CT_num-min_CT_num
    
    #img_data = (img_data - min_CT_num) / (max_CT_num - min_CT_num) * 255
    
    img_data = img_data / (max_CT_num-min_CT_num) * 255 #值域从[0,max_CT_num-min_CT_num]->[0,255]
    print('mha to numpy shape:',img_data.shape)
    weight = img_data.shape[0]
    height = img_data.shape[1]
    channel = img_data.shape[2]
    pad_width = 512 - weight; pad_height = 512 - height
    #判断是否存在文件夹如果不存在则创建为文件夹
    if not os.path.exists(path_png):
        os.makedirs(path_png)

    for i in range(channel):
        img = np.zeros((weight,height), dtype=np.uint8)
        img = img_data[:,:,i]
        #Synergy数据集适用
        #img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)        #逆时针旋转90度
        #TCIA数据集适用
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)        #顺时针旋转90度

        #镜面翻转
        img = cv2.flip(img,1)
        #以0灰度填充图像为512*512
        img = np.pad(img,((0,pad_width),(0,pad_height)),'constant',constant_values=(0,0))
        #保存为png文件
        cv2.imwrite(path_png+'/'+str(i)+'.png',img)
    print('mha shape:',img_data.shape,'to %dpng shape:'%(channel))

def convert_mha_to_npy(mha_file, npy_path):
    # min_CT_num, max_CT_num, len_dicom = mha_read_max_min(mha_file)   #悉尼数据集适用
    # image[image > max_CT_num] = max_CT_num; image[image < min_CT_num] = min_CT_num

    #如果文件名为CBCT的mha文件，则最大值为3000，TCIA的数据集适用
    if 'CBCT' in mha_file:
        max_CT_num = 1000; min_CT_num = -2000
    else:
        max_CT_num = 500; min_CT_num = -1000
    print(min_CT_num,max_CT_num)
    #读取mha文件
    image = sitk.ReadImage(mha_file)
    #将image中大于max_CT_num的值和小于min_CT_num的值赋值为min_CT_num
    #image = sitk.Threshold(image, min_CT_num, max_CT_num, min_CT_num)
    #重构方向
    #image = sitk.PermuteAxes(image, [0,2,1])   #变换mha的方向，从[H,D,W]变成[H,W,D]
    print('orgin mha shape',image.GetSize())
    img_data = sitk.GetArrayFromImage(image).transpose((2, 1, 0))  #转换成numpy，重构方向
    #归一从0到最大值
    img_data = img_data - min_CT_num
    #截取0到(max_CT_num-min_CT_num)
    if 'CBCT' in mha_file:
        img_data[img_data<0] = 0;  img_data[img_data>(max_CT_num-min_CT_num+100)] = 0
    else:
        img_data[img_data<0] = 0; 
    img_data[img_data>(max_CT_num-min_CT_num)] = max_CT_num-min_CT_num
    img_data = img_data / (max_CT_num-min_CT_num) * 2500 #值域从[0,max_CT_num-min_CT_num]->[0,2500]
    print('mha to numpy shape:',img_data.shape)
    weight = img_data.shape[0]
    height = img_data.shape[1]
    channel = img_data.shape[2]
    pad_width = 512 - weight; pad_height = 512 - height
    #判断是否存在文件夹如果不存在则创建为文件夹
    if not os.path.exists(npy_path):
        os.makedirs(npy_path)

    for i in range(channel):
        img = np.zeros((weight,height), dtype=np.float32)
        img = img_data[:,:,i]
        #Synergy数据集适用
        #img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)        #逆时针旋转90度
        #TCIA数据集适用
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)        #顺时针旋转90度
        
        #镜面翻转
        img = cv2.flip(img,1)
        #以0灰度填充图像为512*512
        img = np.pad(img,((0,pad_width),(0,pad_height)),'constant',constant_values=(0,0))
        #保存为npy文件
        np.save(npy_path+'/'+str(i)+'.npy',img)
    print('mha shape:',img_data.shape,'to %dnpy shape:'%(channel))

def mha_to_direct(input_mha,output_mha):
    # 读取第一个.mha图像
    image = sitk.ReadImage(input_mha,outputPixelType=sitk.sitkFloat32)
    #变换mha的方向，从[H,D,W]变成[H,W,D]
    out_image = sitk.PermuteAxes(image, [0,2,1])

    #保存为mha文件
    sitk.WriteImage(out_image, output_mha)
    #打印维度变化
    print('from',image.GetSize(),'to',out_image.GetSize())

def command_iteration(filter):
    print(f"{filter.GetElapsedIterations():3} = {filter.GetMetric():10.5f}")

def mha_to_equal(moving_mha,fixed_mha,output_mha):
    # 读取第一个.mha图像
    fixed_image  = sitk.ReadImage(fixed_mha,outputPixelType=sitk.sitkFloat32)
    moving_image = sitk.ReadImage(moving_mha,outputPixelType=sitk.sitkFloat32)
    # 获取fixed_image的大小和原点
    fixed_size = fixed_image.GetSize()    ;moving_size = moving_image.GetSize()
    fixed_origin = fixed_image.GetOrigin();moving_origin = moving_image.GetOrigin()
    # 计算fixed_image的中心
    fixed_center = [int(fixed_origin[i] + fixed_size[i] / 2) for i in range(len(fixed_size))]  ;moving_center = [moving_origin[i] + moving_size[i] / 2 for i in range(len(moving_size))]
    print(fixed_center,moving_center)
    # 创建配准对象
    registration_method = sitk.ImageRegistrationMethod()

    # 设置配准方法和参数
    #registration_method.SetMetricAsMeanSquares()
    #registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=0.1, minStep=1e-4, numberOfIterations=1)#只迭代一次
    registration_method.SetMetricAsCorrelation()
    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-4,
        numberOfIterations=200,
        gradientMagnitudeTolerance=1e-8,
    )
    registration_method.SetInitialTransform(sitk.TranslationTransform(fixed_image.GetDimension()))


    # 执行配准
    transform = registration_method.Execute(fixed_image, moving_image)

    # 应用变换到移动图像
    registered_image = sitk.Resample(moving_image, fixed_image, transform, sitk.sitkLinear, -1000.0)   #默认填充是0，这里填充-1000

    # 创建一个ROI
    roi_size = [50, 50, 50]  # ROI的大小
    roi_center = fixed_center #[0, 0, 0]  # ROI的中心
    roi = sitk.Image(roi_size, sitk.sitkUInt8)
    roi.TransformIndexToPhysicalPoint(roi_center)
    # 设置ROI
    registration_method.SetMetricFixedMask(roi)

    registration_method.SetMetricAsCorrelation()
    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-4,
        numberOfIterations=5,
        gradientMagnitudeTolerance=1e-8,
    )
    registration_method.SetInitialTransform(sitk.TranslationTransform(fixed_image.GetDimension()))


    # 执行配准
    transform = registration_method.Execute(registered_image, moving_image)

    # 应用变换到移动图像
    registered_image = sitk.Resample(registered_image, fixed_image, transform, sitk.sitkLinear, 0.0)

    # # 保存配准后的图像，确保和fixed_image具有相同的原点、方向和间距和维度
    sitk.WriteImage(registered_image, output_mha)
    #打印维度变化
    print('from',moving_image.GetSize(),'to',registered_image.GetSize())

if __name__ == '__main__':


    # mha_path = r'E:\dataset\temp_mha\P1\direct\CBCTp1.mha'
    # a,b,c =mha_read_max_min(mha_path)
    # print(a,b,c)
    
    # mha_resample(mha_path)
    # mha_normalization(mha_path)
    # g,h,i =mha_read_max_min(r'E:\dataset\2018sydney\P1\MC_T_P1_NS\FDKGroundTruth\FDK3D.mha')
    # print(g,h,i)
    # moving_mha = r'E:\dataset\temp_dicom\100HM10395\CTp1.mha'
    # fixed_mha = r'E:\dataset\temp_dicom\100HM10395\CBCTp1.mha'
    # output_mha = r'E:\dataset\temp_dicom\100HM10395\CTp1_equal.mha'
    # mha_to_equal(moving_mha,fixed_mha,output_mha)
    #mha_to_direct(output_mha,output_mha)
    mha_file =  r'E:\dataset\temp_dicom\100HM10395\CBCTp1.mha'#output_mha 
    convert_mha_to_npy(mha_file, r'E:\dataset\temp_dicom\100HM10395\CBCTp1_mha_npy')
