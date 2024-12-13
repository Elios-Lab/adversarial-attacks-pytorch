import cv2
import os
import numpy as np
import tqdm

# Define directories
input_dir = '/home/pigo/Desktop/BoschDriveU5px/test_100/pixle_110res_35miter_old_res/images/'  # Directory where noisy images are located
#input_dir= 'test'
output_base_dir = '/home/pigo/Desktop/BoschDriveU5px/test_100/pixle_110res_35miter_old_res/'  # Base directory to store filtered images
#output_base_dir='test'

# Kernel sizes to be used for median filtering
kernel_sizes = [3]

# Create output directories for each kernel size
for ksize in kernel_sizes:
    output_dir = os.path.join(output_base_dir, f'filtered_{ksize}x{ksize}/images')
    os.makedirs(output_dir, exist_ok=True)

# Iterate through each image in the input directory
for image_name in tqdm.tqdm(os.listdir(input_dir)):         

    # Load the image
    img_path = os.path.join(input_dir, image_name)

    img = cv2.imread(img_path)

    # Apply median filtering for each kernel size
    for ksize in kernel_sizes:
        filtered_img = cv2.medianBlur(img, ksize)

        # Save the filtered image in the corresponding folder
        output_dir = os.path.join(output_base_dir, f'filtered_{ksize}x{ksize}/images')
        output_img_path = os.path.join(output_dir, image_name)
        cv2.imwrite(output_img_path, filtered_img)

print("Processing complete. Filtered images saved in respective folders.")