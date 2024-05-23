import pydicom
from PIL import Image
import os
import numpy as np


def convert_jpg_to_dicom(jpg_file, dicom_file):
    # Read the JPG image
    image = Image.open(jpg_file)
    image_array = np.asarray(image)
    # Create a new DICOM dataset
    ds = pydicom.Dataset()
    ds.file_meta = pydicom.Dataset()
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    # Set DICOM attributes
    ds.PatientName = "Anonymous"
    ds.PatientID = "123456"
    ds.Modality = "CT"
    ds.Rows, ds.Columns = image.size
    ds.PixelData = image_array.astype('uint16').tobytes()

    # Save the DICOM file
    ds.save_as(dicom_file)



if __name__ == '__main__':
    jpg_path = r'C:\Users\DL\Desktop\lgt_jpg\1.2.840.113619.2.289.3.279711525.550.1710549787.42'
    dicom_path = jpg_path.replace('jpg', 'dcm')
    if not os.path.exists(dicom_path):
        os.makedirs(dicom_path)
    for jpg_file in os.listdir(jpg_path):
        jpg_file_path = os.path.join(jpg_path, jpg_file)
        dicom_file_path = jpg_file_path.replace('jpg', 'dcm')
        convert_jpg_to_dicom(jpg_file_path, dicom_file_path)
        print('JPG to DICOM is done!\n %s has write'%(dicom_file_path))
