from skimage.metrics import peak_signal_noise_ratio 
from skimage.metrics import structural_similarity as ssim1
from skimage.metrics import mean_squared_error

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

    ssim_recon = ssim1(recon0, recon1,
                       data_range=recon0.max() - recon0.min(), win_size=win_size)

    psnr_recon = peak_signal_noise_ratio(recon0, recon1,
                                         data_range= recon0.max() - recon0.min())#recon0.max() - recon1.min())  #data_range的大小影响PSNR的大小

    if verbose:
        err_string = 'MSE: {:.8f}, SSIM: {:.3f}, PSNR: {:.3f}'
        print(err_string.format(mse_recon, ssim_recon, psnr_recon))
    return (mse_recon, ssim_recon, psnr_recon)



if __name__ == '__main__':
    recon0_path = r'E:\dataset\temp_npy\2021121308\CTp6\10.npy'
    recon1_path = r'E:\dataset\temp_npy\2021121308\CBCTp6\10.npy'
    recon0 = np.load(recon0_path)
    print(recon0.max(),recon0.min())
    recon1 = np.load(recon1_path)
    print(recon1.max(),recon1.min())
    mse_recon, ssim_recon, psnr_recon = compare(recon0, recon1) 
    print(recon0_path,mse_recon, ssim_recon, psnr_recon)