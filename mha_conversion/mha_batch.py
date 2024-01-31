import SimpleITK as sitk
import numpy as np
from mha_conversion import *
import os

def mha_batch_direct(mha_path):
    mha_list = os.listdir(mha_path)
    #判断是否存在文件夹如果不存在则创建为文件夹
    if not os.path.exists(os.path.join(mha_path,'direct')):
        os.makedirs(os.path.join(mha_path,'direct'))
    output_path = os.path.join(mha_path,'direct')
    for mha in mha_list:
        #判断是否为文件
        if not os.path.isfile(os.path.join(mha_path,mha)):
            continue
        input_mha  = os.path.join(mha_path,mha)
        output_mha = os.path.join(output_path,mha)
        mha_to_direct(input_mha,output_mha)



if __name__ == '__main__':
    # input_mha = r'E:\dataset\2018sydney\P3\MC_T_P3_LD\FDKRecon\FDK3D.mha'
    # fixed_mha = r'E:\dataset\2018sydney\P3\MC_T_P3_Prior\CT_01.mha'
    # output_mha = r'E:\dataset\temp_mha\P3\CBCTprior.mha'
    # mha_to_equal(input_mha,fixed_mha,output_mha)
    # input_mha = r'E:\dataset\temp_mha\P2\direct\CT_01.mha'
    # output_png= r'E:\dataset\temp_mha\P2\png'
    input_mha = r'E:\dataset\temp_mha\P1\direct\CBCTp1.mha'
    output_png= r'E:\dataset\temp_mha\P1\png1'
    convert_mha_to_png(input_mha,output_png)
    # mha_path = r'E:\dataset\temp_mha\P1'
    # mha_batch_direct(mha_path)