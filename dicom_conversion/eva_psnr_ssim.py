import cv2
#导入峰值信噪比模块
from skimage.metrics import peak_signal_noise_ratio as PSNR
#导入结构相似度模块
from skimage.metrics import structural_similarity as SSIM

import numpy as np

def compare(recon0, recon1, verbose=True):
    #mse_recon = mean_squared_error(recon0, recon1)
    #均方根
    mse_recon = np.sqrt(np.mean((recon0-recon1)**2))
    # np.mean((recon0-recon1)**2)

    small_side = np.min(recon0.shape)
    if small_side < 7:
        if small_side % 2:  # if odd
            win_size = small_side
        else:
            win_size = small_side - 1
    else:
        win_size = None

    ssim_recon = SSIM(recon0, recon1,
                       data_range = recon0.max() - recon0.min(), win_size=win_size)
    #recon0.max() - recon0.min()
    psnr_recon = PSNR(recon0, recon1,
                        data_range=recon0.max() - recon0.min())

    if verbose:
        err_string = 'MSE: {:.8f}, SSIM: {:.3f}, PSNR: {:.3f}'
        print(err_string.format(mse_recon, ssim_recon, psnr_recon))
    return (mse_recon, ssim_recon, psnr_recon)



if __name__ == '__main__':

    recon0 = np.load(r'E:\dataset\temp_npy\119HM10395\CBCTp5\15.npy')
    recon1 = np.load(r'E:\dataset\temp_npy\119HM10395\CTp5\15.npy')
    mse_recon, ssim_recon, psnr_recon = compare(recon0, recon1) 
    print(mse_recon, ssim_recon, psnr_recon)
    cbct_image = r'E:\dataset\temp_png\119HM10395\CBCTp5\15.png'
    ct_image =  r'E:\dataset\temp_png\119HM10395\CTp5\15.png'
    #读取CBCT图像
    cbct_image = cv2.imread(cbct_image)
    #读取CT图像
    ct_image = cv2.imread(ct_image)
    mse_recon, ssim_recon, psnr_recon = compare( cbct_image,ct_image)
    print(mse_recon, ssim_recon, psnr_recon)
