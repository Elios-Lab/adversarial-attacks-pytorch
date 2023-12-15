import os

# Assuming hypothetical directory paths for demonstration
# In practice, replace the following lines with the actual paths to your images and labels
images_path = "/home/elios/small_set/images"
labels_path = "/home/elios/small_set/labels"

# Get the filenames without extensions
image_names = {os.path.splitext(filename)[0] for filename in os.listdir(images_path)}
label_names = {os.path.splitext(filename)[0] for filename in os.listdir(labels_path)}

# Get the intersection of the two sets of filenames
common_names = image_names & label_names

# Filter the files in both directories
for directory in [images_path, labels_path]:
    for filename in os.listdir(directory):
        if os.path.splitext(filename)[0] not in common_names:
            os.remove(os.path.join(directory, filename))