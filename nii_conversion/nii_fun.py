import nibabel as nib

# Specify the path to the NIfTI file
file_path = r"E:\dataset\temp_dicom\100HM10395\nii\CBCTAVG.nii"

# Load the NIfTI file
nii_img = nib.load(file_path)

# Get the shape of the NIfTI data
shape = nii_img.shape

# Print the shape
print(shape)
