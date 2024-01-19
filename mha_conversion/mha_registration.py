import SimpleITK as sitk
import numpy as np

# 读取第一个.mha图像
fixed_image = sitk.ReadImage("CT_01.mha",outputPixelType=sitk.sitkFloat32)

# 读取第二个.mha图像

#convert_mha_direction("FDK4D_01.mha", "FDK4D_01_out.mha")
moving_image = sitk.ReadImage("FDK4D_01.mha",outputPixelType=sitk.sitkFloat32)

# 创建配准对象
registration_method = sitk.ImageRegistrationMethod()

# 设置配准方法和参数
registration_method.SetMetricAsMeanSquares()
registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=0.1, minStep=1e-4, numberOfIterations=100)
registration_method.SetInitialTransform(sitk.TranslationTransform(fixed_image.GetDimension()))

# 执行配准
transform = registration_method.Execute(fixed_image, moving_image)

# 应用变换到移动图像
registered_image = sitk.Resample(moving_image, fixed_image, transform, sitk.sitkLinear, 0.0)

# 保存配准后的图像，确保和fixed_image具有相同的原点、方向和间距和维度
sitk.WriteImage(registered_image, "registered_image.mha")
