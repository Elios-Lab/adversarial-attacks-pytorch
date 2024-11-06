import os
from tqdm import tqdm

def remove_theta_suffix(filename):
    """Remove the _theta and anything after it but keep the file extension."""
    # Split filename into name and extension
    name, ext = os.path.splitext(filename)
    
    # If _theta is in the name, remove everything after _theta
    if "_Pixle" in name:
        name = name.split("_Pixle")[0]
        
    # Return the base name with the original extension
    return name + ".png"

def delete_matching_files(query_folder, target_folder):
    """Delete files in the target folder that match the base name of files in the query folder."""
    # List all files in the query folder
    query_files = os.listdir(query_folder)
    
    # For each file in the query folder
    for query_file in tqdm(query_files):
        # Get the base name by removing the _theta... part, keeping the extension
        base_name_with_ext = remove_theta_suffix(query_file)
        # Look for a matching file in the target folder
        for target_file in os.listdir(target_folder):
            if base_name_with_ext==target_file or ":Zone.Identifier" in target_file:
                # If a matching file is found, delete it
                target_file_path = os.path.join(target_folder, target_file)
                print(f"Deleting {target_file_path}")
                os.remove(target_file_path)
                
# Example usage:
query_folder = '/home/elios/pighetti/adversarial-attacks-pytorch/yolo_adv/yolo_adv/adv_data/adv/adv_img/Pixle/images'
target_folder = '/mnt/d/BoschDriveU5px/train/images'

delete_matching_files(query_folder, target_folder)