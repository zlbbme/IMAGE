import os

folder_path = '/path/to/folder'  # Replace with the actual folder path

# Get the list of file names in the folder
file_names = os.listdir(folder_path)

# Iterate over the file names
for file_name in file_names:
    # Check if the file name starts with a number from 0 to 14
    if file_name.startswith(tuple(str(i) for i in range(15))):
        # Delete the file
        os.remove(os.path.join(folder_path, file_name))
    else:
        # Rename the file with a new name starting with 1
        new_file_name = '1' + file_name[1:]
        os.rename(os.path.join(folder_path, file_name), os.path.join(folder_path, new_file_name))
