import datetime

from BM3D_core import *
from utils import *


def step1_search_block_2d(image: np.ndarray,
                          coordinates: Union[tuple, list, np.ndarray],
                          block_size: int,
                          search_step: int,
                          search_range: int,
                          threshold: Union[int, float],
                          group_size: int,
                          transform: lambda x: x,
                          filter_threshold: Union[int, float] = 0):
    """
    Search for similar blocks in the search range

    Parameters
    ----------
    image : ndarray
        Image to search in
    coordinates : tuple
        Coordinates of the block in the image
    block_size : int
        Size of the block
    search_step : int
        Step of the search
    search_range : int
        Search range
    threshold : Union[int, float]
        Threshold for the difference between blocks
    group_size : int
        Size of the group of similar blocks
    transform : function
        Transform to apply to the block
    filter_threshold : Union[int, float]
        Threshold for the filter applied before distance calculation

    Returns
    -------
    ndarray
        Array of similar blocks
    """

    # get coordinates of the block
    i, j = coordinates

    # get vicinity as reference
    reference = image[i:i + block_size, j:j + block_size]

    transformed_reference = transform(reference)
    # draw reference
    # plt.imshow(reference[0, ..., 0], cmap='gray')
    # plt.axis('off')
    # plt.show()

    # search for similar patches
    blocks = []
    transformed_blocks = []
    for x in range(i - search_range, i + search_range + 1, search_step):
        for y in range(j - search_range, j + search_range + 1, search_step):

            # check if the block is in the image
            if (0 <= x < image.shape[0] - block_size + 1
                    and 0 <= y < image.shape[1] - block_size + 1):

                # get block
                block = image[x:x + block_size, y:y + block_size]

                # transformation
                transformed_block = transform(block)
                # print(np.abs(transformed_block).max(), np.abs(transformed_block).min())

                # apply filter
                transformed_block[np.abs(transformed_block) < filter_threshold] = 0
                transformed_reference[np.abs(transformed_reference) < filter_threshold] = 0

                # calculate distance
                distance = np.mean((transformed_block - transformed_reference) ** 2)
                if distance < threshold:
                    blocks.append(block)
                    transformed_blocks.append(transformed_block)

    # sort blocks by similarity
    blocks = np.array(blocks)
    transformed_blocks = np.array(transformed_blocks)
    distances = np.mean((transformed_blocks - transformed_reference) ** 2, axis=(1, 2))
    # print('dist', distances.max(), distances.min())
    sorted_indices = np.argsort(distances)
    blocks = blocks[sorted_indices]

    # return the first group_size blocks
    try:
        return blocks[:group_size]
    except IndexError:
        return blocks


# load data and initialization
data_path = r'data/CT noise simulation/4DCT/d=1, s=1600.npy'
last_name = data_path.split('/')[-1]
reference_path = r'data/processed DICOM/4DCT.npy'

reference = np.load(reference_path).astype(float)
data = np.load(data_path)
max_val, min_val = data.max(), data.min()

# normalize data
data = (data - min_val) / (max_val - min_val)
reference = (reference - min_val) / (max_val - min_val)
print(data.max(), data.min())

# show image
original_plane = reference[0, ..., 0]
noisy_plane = data[0, ..., 0]
shape = original_plane.shape

# noise estimation
noise_estimator = NoiseEstimator()
noise_estimator.set_image(data)
noise_estimator.set_sampling_size((1, 50, 50, 1))
noise_estimator.set_sampling_coordinates([
    [0, 10, 246, 0],
    [2, 446, 246, 10],
    [4, 75, 80, 20],
    [6, 68, 350, 30],
])
sigma = noise_estimator.sigma
print(f' estimated sigma={sigma}')

# parameters for BM3D Step 1
block_size_hard = 8
move_step = 3
search_step_hard = 1
search_range_hard = 11
distance_threshold_hard = 0.1
transform = dctn
filter_threshold_hard = 2 * sigma
threshold_hard = 3 * sigma
n_hard = 4

# Kaiser window
beta = 2.0
k = np.kaiser(block_size_hard, beta)
kaiser = np.outer(k, k)
# plt.imshow(kaiser, cmap='gray')
# plt.show()

# -------- BM3D Step 1 --------
coordinates = np.array(np.meshgrid(
    np.arange(0, shape[0], move_step),
    np.arange(0, shape[1], move_step),
    indexing='ij'
)).T.reshape(-1, 2)

# padding
padded_plane = np.pad(noisy_plane, (np.array((search_range_hard, search_range_hard)),
                                    (search_range_hard, search_range_hard)),
                      'edge')

basic_img = np.zeros(padded_plane.shape)
weight_img = np.ones(padded_plane.shape)

start = datetime.datetime.now()
start_str = start.strftime('%y-%m-%d %H:%M:%S')
print(f'BM5D Step 1 started, time: {start_str}')

for i, j in coordinates:
    i1, j1 = np.array([i, j]) + search_range_hard  # real coordinates in padded image

    # find similar patches
    block = step1_search_block_2d(padded_plane,
                                  (i1, j1),
                                  block_size_hard,
                                  search_step_hard,
                                  search_range_hard,
                                  distance_threshold_hard,
                                  n_hard,
                                  transform,
                                  filter_threshold_hard)

    # transform
    block_dct = dctn(block, axes=(1, 2))
    transformed_block, cd = hdwtn(block_dct, axes=[0])
    # print(np.abs(transformed_block).max(), np.abs(transformed_block).min())

    # hard thresholding
    filtered_block_dct, numb_nonzeros = apply_threshold(transformed_block, threshold_hard)
    if numb_nonzeros == 0:
        weight = 1
    else:
        weight = 1 / numb_nonzeros

    # weight *= kaiser

    # inverse transform
    block = idctn(ihdwtn(filtered_block_dct, axes=[0], cd=cd), axes=(1, 2))
    average_block = np.mean(block, axis=0)
    # draw average block
    # plt.imshow(average_block[0, ..., 0], cmap='gray')
    # plt.axis('off')
    # plt.show()

    # add estimated block to basic image
    basic_img[i1:i1 + block_size_hard, j1:j1 + block_size_hard] \
        += average_block * weight
    weight_img[i1:i1 + block_size_hard, j1:j1 + block_size_hard] \
        += weight

    # print time
    now = datetime.datetime.now()
    if (now - start).seconds >= 60:
        start = now
        now_str = now.strftime('%y-%m-%d %H:%M:%S')
        print(f'Block: {i}, {j}, time: {now_str}')

# get basic image
# print(weight_img.max(), weight_img.min())
basic_img = basic_img[search_range_hard:search_range_hard + shape[0], search_range_hard:search_range_hard + shape[1]]
weight_img = weight_img[search_range_hard:search_range_hard + shape[0], search_range_hard:search_range_hard + shape[1]]
basic_img /= weight_img
print(basic_img.max(), basic_img.min())
# basic_img = (basic_img - basic_img.min()) / (basic_img.max() - basic_img.min())


# save basic image
out_path = f'results/BM3D Step 1/{last_name}'
# np.save(out_path, basic_img)
# print('results saved')
end = datetime.datetime.now()
end_str = end.strftime('%y-%m-%d %H:%M:%S')
print(f'BM5D Step 1 finished, time: {end_str}')

# show basic image
plt.imshow(basic_img, cmap='gray')
plt.axis('off')
plt.title('BM3D basic estimation')
plt.show()
plt.close()

# print results
evaluator = ImageEvaluator(bits=1)
evaluator.set_reference(original_plane)
evaluator.set_image(noisy_plane)
psnr = evaluator.PSNR
ssim = evaluator.SSIM
print(f'before BM3D: psnr={psnr}, ssim={ssim}')

evaluator.set_image(basic_img)
psnr = evaluator.PSNR
ssim = evaluator.SSIM
print(f'after BM3D: psnr={psnr}, ssim={ssim}')
