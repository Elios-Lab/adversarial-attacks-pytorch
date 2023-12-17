import matplotlib.pyplot as plt
import os
import numpy as np

# frames = sorted(os.listdir('/Users/luca/Documents/adv_sequences_atk/seq1/images/'))

# attack_indices = np.zeros(len(frames))
# attack_mapping = {'None': 0, 'PGD': 1, 'F-FGSM': 2, 'VNI-FGSM': 3, 'Pixle': 4, 'DeepFool': 5}
# # Update the array based on image file names
# for i, frame in enumerate(frames):
#     print(frame)
#     for keyword, index in attack_mapping.items():
#         if keyword.lower().replace("-", "") in frame.lower():
#             attack_indices[i] = index
#             print(attack_indices[i], index)
#             break

weights = np.exp(-np.linspace(0, 5, num=200))
# weights = np.linspace(1, 1/200, num=200)
sum = np.sum(weights * np.full_like(weights, True))
sum = sum / 100
print(f'SUM:\t\t\t\t\t\t{sum}\n')
print('0 is most recent frame, 200 (10s) is least recent frame\n')
a100 = np.full_like(weights, False)
a100[:100] = True
print(f'ADV 0-100 (0-5 s):\t\t\t\t{np.sum(weights * a100) / sum}')
a100 = np.full_like(weights, False)
a100[100:] = True
print(f'ADV 100-200 (5-10 s):\t\t\t\t{np.sum(weights * a100) / sum}')
a50 = np.full_like(weights, False)
a50[150:] = True
a50[50:100] = True
print(f'ADV 50-100 150-200 (2.5-5 s 7.5-10 s):\t\t{np.sum(weights * a50) / sum}')
a50 = np.full_like(weights, False)
a50[100:150] = True
a50[:50] = True
print(f'ADV 0-50 100-150 (2.5-5 s 7.5-10 s):\t\t{np.sum(weights * a50) / sum}')
a50 = np.full_like(weights, False)
a50[150:] = True
print(f'ADV 150-200 (7.5-10 s):\t\t\t\t{np.sum(weights * a50) / sum}')

a10 = np.full_like(weights, False)
a10[:10] = True
print(f'ADV 0-10 (0-0.5 s):\t\t\t\t{np.sum(weights * a10) / sum}')
a_alt = np.full_like(weights, False)
a_alt[0] = a_alt[2] = a_alt[4] = a_alt[6] = a_alt[8] = a_alt[10] = a_alt[12] = a_alt[14] = a_alt[16] = a_alt[18] = a_alt[20] = True
print(f'ADV 0-20 (0-1 s) alternate:\t\t\t{np.sum(weights * a_alt) / sum}')


# # Creating plots
# fig, ax = plt.subplots(figsize=(10, 3))
# ax.plot(np.linspace(1, 1/200, num=200), color='blue')
# ax.plot(np.exp(-np.linspace(0, 5, num=200)), color='red')
# ax.plot(np.exp(-np.linspace(0, 3, num=200)), color='green')
# # ax.step(range(len(attack_indices)), attack_indices, where='post', color='orange')
# # ax.set_yticks(list(attack_mapping.values()))
# # ax.set_yticklabels(list(attack_mapping.keys()))
# ax.set_xlabel('Step')
# ax.set_ylabel('Attack Type')
# ax.grid()
# plt.show()


# import os

# def calculate_percentage_of_files_with_keywords(directory, keywords):
#     # Check if the directory exists
#     if not os.path.exists(directory):
#         return "Directory does not exist."

#     # Initialize variables
#     total_files = 0
#     files_with_keywords = 0

#     # Iterate over files in the directory
#     for filename in os.listdir(directory):
#         total_files += 1

#         # Check for each keyword in the file
#         if any(keyword in filename for keyword in keywords):
#             files_with_keywords += 1

#     # Calculate the percentage
#     if total_files > 0:
#         percentage = (files_with_keywords / total_files) * 100
#     else:
#         return "No files found in the directory."

#     return f"Percentage of files containing specified keywords: {percentage:.2f}%"

# # Example usage
# directory_path = "/Users/luca/Documents/adv_sequences_atk/seq2/images/"
# keywords = ["PGD", "FFGSM", "VNIFGSM", "Pixle", "DeepFool"]
# result = calculate_percentage_of_files_with_keywords(directory_path, keywords)
# print(result)

