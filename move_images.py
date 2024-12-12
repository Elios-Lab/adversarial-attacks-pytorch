import os
import shutil
import argparse

def add_images_with_class(input_dir, target_image_dir, class_number, desired_instances):
    images_dir = os.path.join(input_dir, 'images')
    labels_dir = os.path.join(input_dir, 'labels')
    
    if not os.path.exists(target_image_dir):
        os.makedirs(target_image_dir)

    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    instances_added = 0

    for label_file in label_files:
        if instances_added >= desired_instances:
            break
        
        file_path = os.path.join(labels_dir, label_file)
        with open(file_path, 'r') as file:
            lines = file.readlines()
            class_count = sum(1 for line in lines if line.split()[0] == str(class_number))
            
            if class_count > 0:
                image_file = os.path.splitext(label_file)[0] + '.png'
                image_path = os.path.join(images_dir, image_file)
                
                if os.path.exists(image_path):
                    shutil.copy(image_path, os.path.join(target_image_dir, image_file))
                    instances_added += class_count
                    print(f"Copied {image_file} with {class_count} instances of class {class_number}. Total instances added: {instances_added}")
                    if instances_added >= desired_instances:
                        break
                else:
                    print(f"Image file {image_file} does not exist.")
            else:
                print(f"No instances of class {class_number} in {label_file}.")

    print(f"Added {instances_added} instances of class {class_number} to the target directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add images with a specific class to the target directory.")
    parser.add_argument("input_dir", help="Input directory containing 'images' and 'labels' sub-directories.")
    parser.add_argument("target_image_dir", help="Target directory to save the images.")
    parser.add_argument("class_number", type=int, help="Class number (0-4).")
    parser.add_argument("desired_instances", type=int, help="Number of desired instances.")
    
    args = parser.parse_args()
    
    add_images_with_class(args.input_dir, args.target_image_dir, args.class_number, args.desired_instances)