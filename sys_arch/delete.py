import os
import matplotlib.pyplot as plt
import numpy as np

# Assuming a hypothetical directory path containing the images for demonstration
# In practice, replace the following line with the actual path to your images
frames_path = "/path/to/frames"
frames = ["image_pgd.jpg", "image_pgd.jpg", "image_ffgsm.jpg", "image.jpg", "image_pixle.jpg", "image_deepfool.jpg"]  # Example filenames

# Initialize the array to all zeros
attack_indices = np.zeros(len(frames))

# Mapping of keywords to indices
attack_mapping = {'None': 0, 'PGD': 1, 'FGSM': 2, 'F-FGSM': 3, 'VNI-FGSM': 4, 'Pixle': 5, 'DeepFool': 6}

# Update the array based on image file names
for i, frame in enumerate(frames):
    for keyword, index in attack_mapping.items():
        if keyword.lower() in frame.lower():
            attack_indices[i] = index
            break

# Plotting
fig, ax = plt.subplots(figsize=(7.5, 3))
ax.step(range(len(attack_indices)), attack_indices, where='post', color='b')
ax.set_yticks(list(attack_mapping.values()))
ax.set_yticklabels(list(attack_mapping.keys()))
ax.set_xlabel('Step')
ax.set_ylabel('Attack Type')
ax.grid()
plt.show()
