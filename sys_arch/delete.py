import matplotlib.pyplot as plt
import os
import numpy as np

frames = sorted(os.listdir('/Users/luca/Documents/adv_sequences_atk/seq1/images/'))
frames = frames[:20]

attack_indices = np.zeros(len(frames))
attack_mapping = {'None': 0, 'PGD': 1, 'F-FGSM': 2, 'VNI-FGSM': 3, 'Pixle': 4, 'DeepFool': 5}
# Update the array based on image file names
for i, frame in enumerate(frames):
    print(frame)
    for keyword, index in attack_mapping.items():
        if keyword.lower().replace("-", "") in frame.lower():
            attack_indices[i] = index
            print(attack_indices[i], index)
            break


# Creating plots
fig, ax = plt.subplots(figsize=(10, 3))
ax.step(range(len(attack_indices)), attack_indices, where='post', color='orange')
ax.set_yticks(list(attack_mapping.values()))
ax.set_yticklabels(list(attack_mapping.keys()))
ax.set_xlabel('Step')
ax.set_ylabel('Attack Type')
ax.grid()
plt.show()