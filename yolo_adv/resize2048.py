import os
from PIL import Image

# Paths
base_dir = "/home/lazzaroni/adv/datasets/test_183"
images_dir = os.path.join(base_dir, "images")  # Images directory
labels_dir = os.path.join(base_dir, "labels")  # Labels directory
target_width = 1280
target_height = 720

def resize_and_adjust_labels(images_dir, labels_dir, target_width, target_height):
    for image_file in os.listdir(images_dir):
        if image_file.endswith(".png"):  # Process only PNG images
            image_path = os.path.join(images_dir, image_file)
            label_file = image_file.replace(".png", ".txt")
            label_path = os.path.join(labels_dir, label_file)
            
            # Open the image and get its dimensions
            with Image.open(image_path) as img:
                original_width, original_height = img.size

                # Skip images that are already 1280x720
                if original_width == target_width and original_height == target_height:
                    continue

                # Resize the image
                img_resized = img.resize((target_width, target_height))
                img_resized.save(image_path)  # Overwrite the original image
            
            # Adjust labels only if a corresponding label file exists
            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    lines = f.readlines()

                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    cls, x_center, y_center, width, height = map(float, parts)

                    # Adjust coordinates
                    x_center = x_center * original_width / target_width
                    y_center = y_center * original_height / target_height
                    width = width * original_width / target_width
                    height = height * original_height / target_height

                    new_lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

                # Save adjusted labels
                with open(label_path, "w") as f:
                    f.writelines(new_lines)

# Call the function
resize_and_adjust_labels(images_dir, labels_dir, target_width, target_height)
