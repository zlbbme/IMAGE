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
    # recon0_path = r'/Data/SaveBibMip-SX/eva_data/XCAT/npy/Degraded/Phase1/Degraded10.npy'
    # recon1_path = r'/Data/SaveBibMip-SX/eva_data/XCAT/npy/GT/GT_Phase1/GT10.npy'
    # recon0 = np.load(recon0_path)
    # print(recon0.max(),recon0.min())
    # recon1 = np.load(recon1_path)
    # print(recon1.max(),recon1.min())
    # mse_recon, ssim_recon, psnr_recon = compare(recon0, recon1) 
    # print(recon0_path,mse_recon, ssim_recon, psnr_recon)

    slice_num = 74
    phase_mse = [] ;phase_ssim = [] ;phase_psnr = []
    all_mse = []   ;all_ssim = []   ;all_psnr = []
    for i in range (10):
        for j in range(0,slice_num):
            recon0_path = r'/Data/SaveBibMip-SX/eva_data/test_clinic/CBCTp%d/%d.npy' %(i,j+1)
            recon1_path = r'/Data/SaveBibMip-SX/eva_data/test_clinic/CTp%d/%d.npy'   %(i,j)
            recon0 = np.load(recon0_path)
            print(recon0.max(),recon0.min())
            recon1 = np.load(recon1_path)
            #print(recon1.max(),recon1.min())
            mse_recon, ssim_recon, psnr_recon = compare(recon0, recon1) 
            #print(recon0_path,mse_recon, ssim_recon, psnr_recon)
            phase_mse.append(mse_recon)
            phase_psnr.append(psnr_recon)
            phase_ssim.append(ssim_recon)
        all_mse.append(np.mean(phase_mse))
        all_psnr.append(np.mean(phase_psnr))
        all_ssim.append(np.mean(phase_ssim))
        print('phase%d_mse_mean:%.4f'%(i,np.mean(phase_mse)))
        print('phase%d_psnr_mean:%.4f'%(i,np.mean(phase_psnr)))
        print('phase%d_ssim_mean:%.4f'%(i,np.mean(phase_ssim)))
    print(recon0_path)
    print('all_mse_mean:%.4f'%(np.mean(phase_mse)))
    print('all_psnr_mean:%.4f'%(np.mean(phase_psnr)))
    print('all_ssim_mean:%.4f'%(np.mean(phase_ssim)))
        