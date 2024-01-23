import SimpleITK as sitk
import numpy as np

def mha_to_equal(input_mha,fixed_mha,output_mha):
    # 读取第一个.mha图像
    fixed_image = sitk.ReadImage(fixed_mha,outputPixelType=sitk.sitkFloat32)
    #print('fixed_image',fixed_image.GetSize())
    moving_image = sitk.ReadImage(input_mha,outputPixelType=sitk.sitkFloat32)
    #print('moving_image',moving_image.GetSize())
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
    # for i in range(10):
    #     j = i+1
    #     input_mha = r'E:\dataset\2018sydney\P3\MC_T_P3_LD\FDKRecon\FDK4D_'+'%02d'%j +'.mha'
    #     fixed_mha = r'E:\dataset\2018sydney\P3\MC_T_P3_Prior\CT_'+'%02d'%j +'.mha'
    #     output_mha = r'E:\dataset\temp_mha\CBCTp'+'%d'%j +'.mha'
    #     mha_to_equal(input_mha,fixed_mha,output_mha)

    input_mha = r'E:\dataset\2018sydney\P3\MC_T_P3_LD\FDKRecon\FDK3D.mha'
    fixed_mha = r'E:\dataset\2018sydney\P3\MC_T_P3_Prior\CT_01.mha'
    output_mha = r'E:\dataset\temp_mha\P3\CBCTprior.mha'
    mha_to_equal(input_mha,fixed_mha,output_mha)
