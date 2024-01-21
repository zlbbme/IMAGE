import os

import matplotlib.pyplot as plt
import numpy as np

class ImageEvaluator:
    def __init__(self, bits=8):
        self.image = None
        self.reference = None
        self.bits = bits
        self.max_val = 2 ** bits - 1
        self.k1 = 0.01
        self.k2 = 0.03
        self.hu_diff = 0.03
        self.voxel_diff = 3

    def set_image(self, image):
        self.image = image

    def set_reference(self, reference):
        self.reference = reference

    def set_k1_k2(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    def set_bits(self, bits):
        self.bits = bits
        self.max_val = 2 ** bits - 1

    @property
    def MSE(self):
        return np.mean((self.image - self.reference) ** 2)

    @property
    def PSNR(self):
        mse = self.MSE
        return 10 * np.log10(self.max_val ** 2 / mse)

    @property
    def SSIM(self):
        mean_x = np.mean(self.image)
        mean_y = np.mean(self.reference)
        var_x = np.var(self.image)
        var_y = np.var(self.reference)
        cov = np.cov(self.image.flatten(), self.reference.flatten())[0, 1]
        max_val = 2 ** self.bits - 1
        c1 = (self.k1 * max_val) ** 2
        c2 = (self.k2 * max_val) ** 2
        numerator = (2 * mean_x * mean_y + c1) * (2 * cov + c2)
        denominator = (mean_x ** 2 + mean_y ** 2 + c1) * (var_x + var_y + c2)
        return numerator / denominator


class NoiseEstimator:
    def __init__(self):
        self.image = None
        self.sampling_size = None
        self.coordinates = None

    def set_image(self, image):
        self.image = image

    def set_sampling_size(self, sampling_size):
        self.sampling_size = sampling_size

    def set_sampling_coordinates(self, sampling_coordinates):
        self.coordinates = sampling_coordinates

    @property
    def sigma(self):
        average_std = 0
        numb_coordinates = len(self.coordinates)
        dt, dx, dy, dz = self.sampling_size
        for i in range(numb_coordinates):
            point = self.coordinates[i]
            t, x, y, z = point
            img_slice = self.image[t, ..., z]

            # get sample area
            sample = self.image[t:t + dt, x:x + dx, y:y + dy, z:z + dz]
            std = np.std(sample)
            average_std += std

        average_std = average_std / numb_coordinates
        return average_std


def draw_4d_image(image, title):
    try:
        image = np.array(image.get())
    except AttributeError:
        pass
    plt.subplots(4, 4, figsize=(10, 10))
    for i in range(4):
        for l in range(4):
            img_slice = image[i, :, :, l]
            plt.subplot(4, 4, i * 4 + l + 1)
            plt.imshow(img_slice, cmap='gray')
            plt.axis('off')
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.suptitle(title)
    plt.show()
    plt.close()


def draw_subplots(image_dict: dict,
                  names: list[str],
                  rows: int,
                  cols: int,
                  title: str,
                  figsize=(10, 10),
                  save_path=None):
    """
    draw subplots
    Parameters
    ----------
    image_dict: dict
        images to draw
    names: list[str]
        list of image names
    rows: int
        number of rows
    cols: int
        number of columns
    title: str
        title of the figure
    figsize: tuple
        figure size
    save_path: str
        path to save the figure
    """

    if rows * cols < len(image_dict):
        raise ValueError('rows * cols < len(images)')
    plt.figure(figsize=figsize)
    for i in range(len(names)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(image_dict[names[i]], cmap='gray')
        plt.title(names[i])
        plt.axis('off')
    plt.suptitle(title)
    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    if save_path is not None:
        directory = os.path.dirname(save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
        plt.close()


def apply_ww_wl(image, ww, wl):
    """
    apply window width and window level to the image
    Parameters
    ----------
    image: np.ndarray
        image to be processed
    ww: float
        window width
    wl: float
        window level
    Returns
    -------
    np.ndarray
        processed image
    """
    image = image.astype(np.float32)
    image = np.clip(image, wl - ww // 2, wl + ww // 2)
    image = (image - (wl - ww // 2)) / ww * 255
    return image.astype(np.uint8)


def select_device(device='cuda', print_info=False):
    if device in ['cpu', 'CPU']:
        if print_info:
            print('Using CPU')
        import numpy
        return numpy

    elif device in ['cuda', 'CUDA', 'gpu', 'GPU']:
        try:
            import cupy
            if cupy.cuda.is_available():
                if print_info:
                    print('Using CUDA')
                return cupy
            else:
                if print_info:
                    print('CUDA not available, using CPU')
                import numpy
                return numpy
        except ImportError:
            if print_info:
                print('Failure in importing cupy, using CPU')
            import numpy
            return numpy

    else:
        raise ValueError('Unrecognized device')


def send_email(time_now):
    import smtplib
    from email.mime.text import MIMEText

    server = smtplib.SMTP('smtp.qq.com', 587)
    tolist = ['1282057552@qq.com']
    from_addr = '1282057552@qq.com'
    password = 'ivfpthhjfvjxhhch'

    msg = MIMEText(f'Program finished at {time_now}')
    msg['Subject'] = 'Program Finished'
    msg['From'] = 'wt <1282057552@qq.com>'
    msg['To'] = '1282057552@qq.com'
    server.login(from_addr, password)
    server.sendmail(from_addr, tolist, msg.as_string())
    server.quit()


