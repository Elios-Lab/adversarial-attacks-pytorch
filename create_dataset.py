import os
import shutil
import random
import re
from PIL import Image
from tqdm import tqdm

# Paths to the original datasets
clean_path = '/home/elios/lazzaroni/adv/datasets/clean'
pixle_path = '/home/elios/lazzaroni/adv/datasets/pixle'
poltergeist_path = '/home/elios/lazzaroni/adv/datasets/poltergeist'

# Path to the new dataset
new_dataset_path = '/home/elios/lazzaroni/adv/datasets/FC_5K_3K_3K'

# Number of images for each split
num_images = {
    'train': 5000,
    'val': 3000,
    'test': 3000
}

# Splits and classes
splits = ['train', 'val', 'test']
classes = ['clean', 'pixle', 'poltergeist']

# Create new dataset directory structure
def create_directory_structure(base_path, splits, classes):
    for split in splits:
        for cls in classes:
            dir_path = os.path.join(base_path, split, cls)
            os.makedirs(dir_path, exist_ok=True)

create_directory_structure(new_dataset_path, splits, classes)

# Function to get base filenames from clean dataset for a split
def get_base_filenames(path, split, subfolder='images'):
    split_path = os.path.join(path, split)
    if os.path.exists(os.path.join(split_path, subfolder)):
        split_path = os.path.join(split_path, subfolder)
    base_filenames = set()
    for fname in os.listdir(split_path):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            base_name = os.path.splitext(fname)[0]
            base_filenames.add(base_name)
    return base_filenames

# Function to process and copy images
def process_and_copy_images(split, basenames, source_paths, dest_path):
    for base_name in tqdm(basenames, desc=f"Processing {base_name}"):
        for cls, source_path in source_paths.items():
            split_source_path = os.path.join(source_path, split)
            if cls == 'clean' and os.path.exists(os.path.join(split_source_path, 'images')):
                split_source_path = os.path.join(split_source_path, 'images')
            if not os.path.exists(split_source_path):
                continue
            pattern = re.compile('^' + re.escape(base_name))
            matched_files = [f for f in os.listdir(split_source_path)
                             if pattern.match(os.path.splitext(f)[0]) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            random.shuffle(matched_files)
            for fname in tqdm(matched_files, desc=f"Copying files for {base_name} - {cls}", leave=False):
                src_file = os.path.join(split_source_path, fname)
                dest_dir = os.path.join(dest_path, split, cls)
                dest_file = os.path.join(dest_dir, os.path.splitext(fname)[0] + '.png')
                try:
                    with Image.open(src_file) as img:
                        # img = img.convert('RGB')
                        # img = img.resize((640, 480))
                        img.save(dest_file, 'PNG')
                except Exception as e:
                    print(f"Error processing file {src_file}: {e}")

source_paths = {
    'clean': clean_path,
    'pixle': pixle_path,
    'poltergeist': poltergeist_path
}

for split in splits:
    clean_basenames = get_base_filenames(clean_path, split)
    selected_basenames = random.sample(list(clean_basenames), min(num_images[split], len(clean_basenames)))
    process_and_copy_images(split, selected_basenames, source_paths, new_dataset_path)