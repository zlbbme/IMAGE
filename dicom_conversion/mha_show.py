import numpy as np
import matplotlib.pyplot as plt




def npy_show(npy_file):
    np_data = np.load(npy_file)
    print(np_data.shape)
    plt.imshow(np_data,cmap='gray')
    plt.show()


if __name__ == '__main__':  
    # Path to the .npy file
    #两幅图像并列显示
    npy_data1 = np.load(r'E:\dataset\temp_dicom\100HM10395\CBCTp1_mha_npy\10.npy')
    npy_data2 = np.load(r'E:\dataset\temp_dicom\100HM10395\CTp1_mha_npy\10.npy')
    #两幅图像上下显示
    plt.subplot(211)
    plt.imshow(npy_data1,cmap='gray')
    plt.subplot(212)
    plt.imshow(npy_data2,cmap='gray')
    plt.show()