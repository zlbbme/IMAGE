import datetime

from BM3D_core import *
from utils import *


def step2_search_block_2d(image,
                          step1_image,
                          coordinates: Union[tuple, list, np.ndarray],
                          block_size,
                          search_step,
                          search_range,
                          threshold: Union[int, float],
                          group_size):
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

    Returns
    -------
    ndarray
        Array of similar blocks
    """

    # get coordinates of the block
    i, j = coordinates

    # get vicinity as reference
    reference = image[i:i + block_size, j:j + block_size]
    step1_reference = step1_image[i:i + block_size, j:j + block_size]

    # draw reference
    # plt.imshow(reference[0, ..., 0], cmap='gray')
    # plt.axis('off')
    # plt.show()

    blocks = []
    step1_blocks = []

    # search for similar patches
    for x in range(i - search_range, i + search_range + 1, search_step):
        for y in range(j - search_range, j + search_range + 1, search_step):

            # check if the block is in the image
            if (0 <= x < image.shape[0] - block_size + 1
                    and 0 <= y < image.shape[1] - block_size + 1):

                # get block
                block = image[x:x + block_size, y:y + block_size]
                step1_block = step1_image[x:x + block_size, y:y + block_size]

                # calculate distance
                distance = np.mean((step1_block - step1_reference) ** 2)
                # print(distance)
                if distance < threshold:
                    blocks.append(block)
                    step1_blocks.append(step1_block)

    # sort blocks by similarity
    blocks = np.array(blocks)
    step1_blocks = np.array(step1_blocks)
    distances = np.mean((step1_blocks - step1_reference) ** 2, axis=(1, 2))
    # print('dist', distances.max(), distances.min())
    sorted_indices = np.argsort(distances)
    blocks = blocks[sorted_indices]
    step1_blocks = step1_blocks[sorted_indices]

    # return the first group_size blocks
    try:
        return blocks[:group_size], step1_blocks[:group_size]
    except IndexError:
        return blocks, step1_blocks


# load data and initialization
data_path = r'data/CT noise simulation/4DCT/d=1, s=200.npy'
last_name = data_path.split('/')[-1]
step1_data_path = f'results/BM3D Step 1/{last_name}'
reference_path = r'data/processed DICOM/4DCT.npy'

reference = np.load(reference_path).astype(float)
data = np.load(data_path).astype(float)
step1_data = np.load(step1_data_path).astype(float)

# normalize data
max_val, min_val = data.max(), data.min()
data = (data - min_val) / (max_val - min_val)
reference = (reference - min_val) / (max_val - min_val)

# show image
original_plane = reference[0, ..., 0]
noisy_plane = data[0, ..., 0]
step1_plane = step1_data
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

# another noise estimation method
# sigma = np.std(data - step1_data)


# hyperparameters for BM3D Step 2
block_size_wiener = 8
move_step = 3
search_step_wiener = 1
search_range_wiener = 11
distance_threshold_wiener = 0.001
n_wiener = 4

# Kaiser window
beta = 1.5
k = np.kaiser(block_size_wiener, beta)
kaiser = np.outer(k, k)
# plt.imshow(kaiser, cmap='gray')
# plt.show()

# -------- BM3D Step 2 --------
coordinates = np.array(np.meshgrid(
    np.arange(0, shape[0], move_step),
    np.arange(0, shape[1], move_step),
    indexing='ij'
)).T.reshape(-1, 2)

padded_plane = np.pad(noisy_plane, (np.array((search_range_wiener, search_range_wiener)),
                                    (search_range_wiener, search_range_wiener)),
                      'edge')
padded_reference = np.pad(step1_plane, (np.array((search_range_wiener, search_range_wiener)),
                                        (search_range_wiener, search_range_wiener)),
                          'edge')

result_img = np.zeros(padded_plane.shape)
weight_img = np.ones(padded_plane.shape)

start = datetime.datetime.now()
start_str = start.strftime('%y-%m-%d %H:%M:%S')
print(f'BM5D Step 2 started, time: {start_str}')

for i, j in coordinates:
    i1, j1 = np.array([i, j]) + search_range_wiener  # real coordinates in padded image

    # find similar patches
    blocks, step1_blocks = step2_search_block_2d(padded_plane,
                                                 padded_reference,
                                                 (i1, j1),
                                                 block_size_wiener,
                                                 search_step_wiener,
                                                 search_range_wiener,
                                                 distance_threshold_wiener,
                                                 n_wiener)
    # print(f'block shape: {blocks.shape}')
    # empirical wiener filtering
    filttered_blocks, w = wiener_filter(blocks, step1_blocks, sigma)
    # print(filttered_blocks.shape)
    weight = 1 / (w ** 2)
    # print(weight)
    weight *= kaiser

    average_block = np.mean(filttered_blocks, axis=0)
    # draw average block
    # plt.imshow(average_block[0, ..., 0], cmap='gray')
    # plt.axis('off')
    # plt.show()

    # add estimated block to basic image
    result_img[i1:i1 + block_size_wiener, j1:j1 + block_size_wiener] \
        += average_block * weight
    weight_img[i1:i1 + block_size_wiener, j1:j1 + block_size_wiener] \
        += weight

    # print time
    now = datetime.datetime.now()
    if (now - start).seconds >= 60:
        start = now
        now_str = now.strftime('%y-%m-%d %H:%M:%S')
        print(f'Block: {i}, {j}, time: {now_str}')

# normalize
result_img /= weight_img
result_img = result_img[search_range_wiener:-search_range_wiener,
                        search_range_wiener:-search_range_wiener]
print(result_img.max(), result_img.min())
# result_img = (result_img - result_img.min()) / (result_img.max() - result_img.min())

# save basic image
out_path = f'results/BM3D Step 2/{last_name}'
np.save(out_path, result_img)
print('results saved')
end = datetime.datetime.now()
end_str = end.strftime('%y-%m-%d %H:%M:%S')
print(f'BM5D Step 2 finished, time: {end_str}')

# show result
plt.imshow(result_img, cmap='gray')
plt.axis('off')
plt.title('BM3D final estimation')
plt.show()
plt.close()

# print results
evaluator = ImageEvaluator(bits=1)
evaluator.set_reference(original_plane)
evaluator.set_image(noisy_plane)
psnr = evaluator.PSNR
ssim = evaluator.SSIM
print(f'before BM3D: psnr={psnr}, ssim={ssim}')

evaluator.set_image(step1_plane)
psnr = evaluator.PSNR
ssim = evaluator.SSIM
print(f'after BM3D Step 1: psnr={psnr}, ssim={ssim}')

evaluator.set_image(result_img)
psnr = evaluator.PSNR
ssim = evaluator.SSIM
print(f'after BM3D Step 2: psnr={psnr}, ssim={ssim}')
