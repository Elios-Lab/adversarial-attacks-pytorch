import os

# Define the directory and the substring
directory = "/home/elios/pighetti/adversarial-attacks-pytorch/yolo_adv/adv_data/adv/adv_img/VNIFGSM/images/"
substring = "_VNIFGSM"

# Iterate over all files in the directory
for filename in os.listdir(directory):
    # If the substring is in the filename, rename the file
    if substring in filename:
        new_filename = filename.replace(substring, "")
        os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))