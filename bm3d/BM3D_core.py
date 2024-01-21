from pywt import dwt, idwt
from scipy.fftpack import dct, idct


# Discrete cosine transformation and its inverse transformation
def dctn(x, axes=None, norm='ortho'):
    if axes is None:
        for i in range(x.ndim):
            x = dct(x, axis=i, norm=norm)
        return x
    else:
        for axis in axes:
            x = dct(x, axis=axis, norm=norm)
        return x


def idctn(x, axes=None, norm='ortho'):
    if axes is None:
        for i in range(x.ndim):
            x = idct(x, axis=i, norm=norm)
        return x
    else:
        for axis in axes:
            x = idct(x, axis=axis, norm=norm)
        return x


# Discrete Haar wavelet transformation and its inverse transformation
def hdwtn(x, axes=None, mode='symmetric'):
    if axes is None:
        for i in range(x.ndim):
            x = dwt(x, wavelet='haar', axis=i, mode=mode)
        return np.array(x)
    else:
        for axis in axes:
            x = dwt(x, wavelet='haar', axis=axis, mode=mode)
        return np.array(x)


def ihdwtn(x, axes=None, mode='symmetric', cd=None):
    if cd is None:
        cd = np.zeros(x.shape)
    if axes is None:
        for i in range(x.ndim):
            x = idwt(x, wavelet='haar', axis=i, mode=mode)
        return np.array(x)
    else:
        for axis in axes:
            x = idwt(x, wavelet='haar', axis=axis, mode=mode, cD=cd)
        return np.array(x)


def bwtn(x, axes=None, mode='symmetric'):
    if axes is None:
        for i in range(x.ndim):
            x = dwt(x, wavelet='bior2.2', axis=i, mode=mode)
        return np.array(x)
    else:
        for axis in axes:
            x = dwt(x, wavelet='bior2.2', axis=axis, mode=mode)
        return np.array(x)


def ibwtn(x, axes=None, mode='symmetric', cd=None):
    if cd is None:
        cd = np.zeros(x.shape)
    if axes is None:
        for i in range(x.ndim):
            x = idwt(x, wavelet='bior2.2', axis=i, mode=mode)
        return np.array(x)
    else:
        for axis in axes:
            x = idwt(x, wavelet='bior2.2', axis=axis, mode=mode, cD=cd)
        return np.array(x)


def apply_threshold(block, threshold):
    """
    Apply threshold to the image

    Parameters
    ----------
    block : ndarray
        Block to apply threshold to
    threshold : int
        Threshold

    Returns
    -------
    ndarray
        Image with threshold applied
    ndarray
        Weights of the images
    """

    # apply threshold
    block_new = np.array(block).copy()
    block_new[np.abs(block) < threshold] = 0
    # calculate number of nonzero elements
    numb_nonzero = np.sum(block != 0)
    # print(numb_nonzero)

    return block, numb_nonzero


def wiener_filter(blocks, step1_blocks, sigma):
    """
    Apply Wiener filter to the blocks

    Parameters
    ----------
    blocks : ndarray
        Blocks to apply Wiener filter to
    step1_blocks : ndarray
        Blocks from the first step
    sigma : float
        Noise level

    Returns
    -------
    ndarray, ndarray
        Filtered blocks and shrinkage coefficients
    """

    # calculate shrinkage coefficient
    ndim = step1_blocks.ndim
    dims = np.array(range(ndim))[1:]
    transformed_step1_blocks, _ = hdwtn(dctn(step1_blocks, axes=dims), axes=[0])
    norm_2 = np.sum(transformed_step1_blocks ** 2, axis=(0,))
    w = norm_2 / (norm_2 + sigma ** 2)

    # shrinkage
    transformed_blocks, cd = hdwtn(dctn(blocks, axes=dims), axes=[0])
    filtered_blocks = transformed_blocks * w
    # print(transformed_blocks.shape, w.shape, filtered_blocks.shape)

    # inverse transform
    filtered_blocks = idctn(ihdwtn(filtered_blocks, axes=[0], cd=cd), axes=dims)
    return filtered_blocks, w


if __name__ == '__main__':
    import numpy as np
    from utils import ImageEvaluator, draw_4d_image, NoiseEstimator

    # load data and initialization
    noise_level = 1600
    data_path = f'data/CT noise simulation/4DCT/d=1, s={noise_level}.npy'
    reference_path = r'data/processed DICOM/4DCT.npy'
    reference = np.load(reference_path)
    reference = reference - reference.min()
    data = np.load(data_path)
    data = (data - data.min()) / (data.max() - data.min())
    print(data.max(), data.min())

    # estimate noise level
    estimator = NoiseEstimator()
    estimator.set_image(data)
    estimator.set_sampling_size([1, 50, 50, 1])
    estimator.set_sampling_coordinates([
        [0, 10, 246, 0],
        [2, 446, 246, 10],
        [4, 75, 80, 20],
        [6, 68, 350, 30],
    ])
    sigma = estimator.sigma
    print(sigma)
    threshold = 3 * sigma

    # original quality
    evaluator = ImageEvaluator()
    evaluator.set_image(data)
    evaluator.set_reference(reference)
    evaluator.set_bits(16)
    mse = evaluator.MSE
    psnr = evaluator.PSNR
    ssim = evaluator.SSIM
    print(f'before amplitude filter: mse={mse}, psnr={psnr}, ssim={ssim}')
    # draw_4d_image(reference, 'original image')
    # draw_4d_image(data, 'noisy image')

    # amplitude filter
    data_dct = dctn(data)
    # draw_4d_image(np.log10(np.abs(data_dct)), 'dct image')
    data_dct[np.abs(data_dct) < threshold] = 0
    new_data = idctn(data_dct)
    print(new_data.max(), new_data.min())
    new_data = new_data - new_data.min()
    evaluator.set_image(new_data)
    mse = evaluator.MSE
    psnr = evaluator.PSNR
    ssim = evaluator.SSIM
    print(f'after amplitude filter: mse={mse}, psnr={psnr}, ssim={ssim}')
    # draw_4d_image(new_data, 'filtered image')
