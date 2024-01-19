import numpy as np
import matplotlib.pyplot as plt




def npy_show(npy_file):
    np_data = np.load(npy_file)
    print(np_data.shape)
    plt.imshow(np_data,cmap='gray')
    plt.show()


if __name__ == '__main__':  
    # Path to the .npy file
    file_path = 'npy/20.npy'
    npy_show(file_path)