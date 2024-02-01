
import numpy as np
import odl
import pydicom
import scipy.io as sio
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from scipy import io
from PIL import Image
import numpy as np
from skimage import io as ioi
from skimage import color

# Parameters
lam = 0.01
data_matching = 'exact'


space = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20], shape=[512, 512])



angle_partition = odl.uniform_partition(0, np.pi,240)
# Detector: uniformly sampled, n = 512, min = -30, max = 30
detector_partition = odl.uniform_partition(-30, 30, 512)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)


ray_trafo = odl.tomo.RayTransform(space, geometry)


# phantom = odl.phantom.shepp_logan(space, modified=True)
# phantom = np.float32(pydicom.read_file('2755.IMA').pixel_array)
# phantom = phantom/np.max(phantom)
# data = ray_trafo(phantom)
phantom = ioi.imread('./testdata/test.png')
phantom = color.rgb2gray(phantom[:, :, :3])

phantom = phantom.astype(np.float32)
print(phantom.shape)
# phantom = Image.open('./testdata/test.png')
# phantom = np.array(phantom).astype(np.float32)
print(np.min(phantom), np.min(phantom))
phantom = phantom/np.max(phantom)
print(phantom.shape)
data = ray_trafo(phantom)



# Gradient for TV
gradient = odl.Gradient(space)


f = odl.solvers.IndicatorBox(space, 0, 1)

if data_matching == 'exact':
    # Functional to enforce Ax = g
    # Due to the splitting used in the douglas_rachford_pd solver, we only
    # create the functional for the indicator function on g here, the forward
    # model is handled separately.
    indicator_zero = odl.solvers.IndicatorZero(ray_trafo.range)
    indicator_data = indicator_zero.translated(data)
elif data_matching == 'inexact':
    # Functional to enforce ||Ax - g||_2 < eps
    # We do this by rewriting the condition on the form
    # f(x) = 0 if ||A(x/eps) - (g/eps)||_2 < 1, infinity otherwise
    # That function (with A handled separately, as mentioned above) is
    # implemented in ODL as the IndicatorLpUnitBall function.
    # Note that we use right multiplication in order to scale in input argument
    # instead of the result of the functional, as would be the case with left
    # multiplication.
    eps = 5.0


    raw_noise = odl.phantom.white_noise(ray_trafo.range)
    data += raw_noise * eps / raw_noise.norm()


    indicator_l2_ball = odl.solvers.IndicatorLpUnitBall(ray_trafo.range, 2)
    indicator_data = indicator_l2_ball.translated(data / eps) * (1 / eps)
else:
    raise RuntimeError('unknown data_matching')

# TV minimization
cross_norm = lam * odl.solvers.GroupL1Norm(gradient.range)


lin_ops = [ray_trafo, gradient]
g = [indicator_data, cross_norm]


callback = (odl.solvers.CallbackShow('Iterates', step=5, clim=[0, 1]) &
            odl.solvers.CallbackPrintIteration())

# Solve with initial guess x = 0.
# Step size parameters are selected to ensure convergence.
# See douglas_rachford_pd doc for more information.
# initial_array = sio.loadmat('./testdata/5495-Enet.mat')['Enet']
# print(initial_array.shape)
# plt.imshow(initial_array, cmap='gray')
# plt.show()
x = ray_trafo.domain.zero()
# x = ray_trafo.domain.element(initial_array)

odl.solvers.douglas_rachford_pd(x, f, g, lin_ops,
                                tau=0.1, sigma=[0.1, 0.02], lam=1.5,
                                niter=300, callback=callback)


fbp_recon = odl.tomo.fbp_op(ray_trafo)(data)
fbp_recon.show('FBP Reconstruction')
# phantom.show('Phantom')
data.show('Sinogram', force_show=True)



# 如果需要将结果转换为NumPy数组，可以使用以下代码
result_array = x.asarray()
print(result_array.shape)
plt.imshow(result_array, cmap='gray')
plt.show()
# phantom_array = phantom.asarray() #phatom
phantom_array = phantom #real

ssim_score = ssim(phantom_array, result_array)
psnr_score = psnr(phantom_array, result_array)
mse_score = mse(phantom_array, result_array)
io.savemat('Testreuslt/test.mat', {'reference': phantom_array, 'TVMRecon60': result_array})
print('SSIM:', ssim_score)
print('PSNR:', psnr_score)
print('MSE:', mse_score)



def calculate_nmse(y, y_hat):
    return 100*np.sum((y - y_hat) ** 2) / np.sum(y ** 2)

nmse = calculate_nmse(phantom_array, result_array)

print('Normalized Mean Square Error:', nmse)