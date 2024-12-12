import os
import shutil
import argparse

def copy_labels(images_dir, labels_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
    
    for image_file in image_files:
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)
        
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(output_dir, label_file))
        else:
            print(f"Label for {image_file} not found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy labels for images to a new directory.")
    parser.add_argument("--images_dir", help="Directory containing the images.")
    parser.add_argument("--labels_dir", help="Directory containing the labels.")
    parser.add_argument("--output_dir", help="Directory to save the copied labels.")
    
    args = parser.parse_args()
    
    copy_labels(args.images_dir, args.labels_dir, args.output_dir)