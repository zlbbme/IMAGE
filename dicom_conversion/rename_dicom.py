import os
import pydicom

def rename_dicom_files(directory):
    #读取文件夹内文件个数
    file_nums = len(os.listdir(directory))
    print(file_nums)
    for filename in os.listdir(directory):
        if filename.endswith(".DCM"):
            filepath = os.path.join(directory, filename)
            ds = pydicom.dcmread(filepath, force=True)
            ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
            instance_number = ds.InstanceNumber
            new_filename = f"{file_nums-instance_number}.dcm"
            print(new_filename)
            new_filepath = os.path.join(directory, new_filename)
            os.rename(filepath, new_filepath)

# Usage example
for i in range (10):
    directory = r"E:\dataset\test_clinic\Fraction3\CBCTp"+str(i)
    rename_dicom_files(directory)

prior_directory = r"E:\dataset\test_clinic\Fraction3\CBCTpriorAVG"
rename_dicom_files(prior_directory)

