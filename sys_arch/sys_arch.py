from yolo_adv.classifier.classifier import Classifier
from ultralytics import YOLO
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
from tqdm import tqdm

class FrameSequence:
    def __init__(self, window_size, frames_path):
        self.window_size = window_size
        self.frames = sorted(os.listdir(frames_path))
        self.num_of_frames = len(self.frames)
        self.adversarial_detections = np.full(window_size, False)
        # self.weights = np.linspace(1, 1 / self.window_size, num=self.window_size)
        self.weights = np.exp(-np.linspace(0, 5, num=self.window_size))
        self.worst_case = np.sum(self.weights * np.full_like(self.weights, True))
        # Initialize the array of attacks to all zeros
        self.attack_indices = np.zeros(len(self.frames))
        self.attack_mapping = {'None': 0, 'PGD': 1, 'F-FGSM': 2, 'VNI-FGSM': 3, 'Pixle': 4, 'DeepFool': 5}
        # Update the array based on image file names
        for i, frame in enumerate(self.frames):
            for keyword, index in self.attack_mapping.items():
                if keyword.lower().replace("-", "") in frame.lower():
                    self.attack_indices[i] = index
                    break
     
    def update_detections(self, is_adversarial):
        # Add the latest detection and ensure the list size doesn't exceed the window size
        self.adversarial_detections = np.roll(self.adversarial_detections, 1)
        self.adversarial_detections[0] = is_adversarial
                
    def calculate_weighted_sum(self):
        # Apply linearly increasing weights and calculate the weighted sum
        return np.sum(self.weights * self.adversarial_detections) / self.worst_case * 100 # Normalize to 100
    
class DecisionMaker:
    def __init__(self, threshold):
        self.threshold = threshold

    def check_for_tor(self, ws):
        # Check if the weighted sum exceeds the threshold
        if ws > self.threshold:
            return True  # Trigger TOR
        else:
            return False  # Continue normal execution
        
    def check_for_ad(self, ws):
        # Check if the weighted sum has become lower than the threshold
        if ws <= self.threshold:
            return True  # Enable automated driving mode
        else:
            return False  # Continue manual driving mode
        
def compress_image_to_jpeg(image, quality=100):
    # Create a BytesIO object to hold the compressed image
    in_memory_file = io.BytesIO()

    # Save the image to the in-memory file with specified quality
    image.save(in_memory_file, format='JPEG', quality=quality)
    in_memory_file.seek(0)  # Seek to the start of the file
    image = Image.open(in_memory_file)

    # Return the in-memory file
    return image

# def true_jpeg_compression(image, perc=40):
#     # Simulate a JPEG compression that is able to remove adversarial perturbations 40% of the time
#     return np.random.randint(0, 100) > perc
        
if __name__ == "__main__":
    window_size = 200
    to_threshold = 12
    ad_threshold = 0
    frames_path = '/home/elios/Desktop/adv_sequence_cam_0_600_atk/seq2/images/'
    fs = FrameSequence(window_size, frames_path=frames_path)
    classifier = Classifier()
    classifier.load_model(model_path='yolo_adv/classifier/cls_last.pt')
    tld_model = YOLO('yolo_adv/best.pt')
    todm = DecisionMaker(threshold=to_threshold)
    adm = DecisionMaker(threshold=ad_threshold)
    automated_mode = True  # Start in automated driving mode
    
    fc_history = np.full(fs.num_of_frames, False)  # True if frame is classified as clean
    fe_history = np.full(fs.num_of_frames, False)  # True if frame is enhanced
    ad_history = np.full(fs.num_of_frames, False)  # True while in AD mode
    thrs_history = np.full(fs.num_of_frames, 0, dtype=np.float32)  # Threshold history
    
    for i, frame in tqdm(enumerate(fs.frames), total=fs.num_of_frames, unit='frame'):
        image = Image.open(frames_path+frame)
        is_adversarial = classifier.predict(image=image) == 'adv'  
        if is_adversarial:  # Try frame enhancement
            image = compress_image_to_jpeg(image, quality=100)
            is_adversarial = classifier.predict(image=image) == 'adv'
            # is_adversarial = true_jpeg_compression(image, perc=46)
            fe_history[i] = True
        fc_history[i] = is_adversarial
        fs.update_detections(is_adversarial)
        threshold = fs.calculate_weighted_sum()
        thrs_history[i] = threshold
        if not is_adversarial:  # If the frame is not adversarial, run TLD
            # results = tld_model.predict(image)
            if not automated_mode:  # Driving in manual mode
                automated_mode = ad_history[i] = adm.check_for_ad(threshold)
            # TODO else output detection results
            else:  # Driving in AD mode and no adv detected, so continue AD
                automated_mode = ad_history[i] = True
        else: # Frame is adversarial, enhancement not successful
            if not automated_mode: # Driving in manual mode
                automated_mode = ad_history[i] = adm.check_for_ad(threshold)
            else:  # Driving in AD mode
                automated_mode = ad_history[i] = not todm.check_for_tor(threshold)  # If TOR is triggered, switch to manual mode
                
    
    # Convert boolean arrays to integers
    fc_history = fc_history.astype(int)
    fe_history = fe_history.astype(int)
    ad_history = np.logical_not(ad_history).astype(int)
    
    figsize = (10, 4)

    # Creating plots
    fig0, ax0 = plt.subplots(figsize=figsize)
    ax0.step(range(len(fs.attack_indices)), fs.attack_indices, where='post', color='orange')
    ax0.set_yticks(list(fs.attack_mapping.values()))
    ax0.set_yticklabels([key.replace('F-', '') if key.startswith('F-') else key for key in fs.attack_mapping.keys()])
    ax0.set_xlabel('Step')
    ax0.set_ylabel('Attack Type')
    ax0.grid()
    # plt.show()
    
    fig1, ax1 = plt.subplots(figsize=figsize)
    ax1.step(range(len(thrs_history)), thrs_history, where='post', color='purple')
    # ax1.plot(thrs_history, color='yellow')
    ax1.set_title('Threshold History')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Value')
    ax1.grid()
    ax1.set_ylim([-0.1, 100.1])
    ax1.axhline(y=to_threshold, linestyle='--', color='red')
    ax1.axhline(y=ad_threshold, linestyle='--', color='green')
    # plt.show()

    fig2, ax2 = plt.subplots(figsize=figsize)
    ax2.step(range(len(fc_history)), fc_history, where='post', color='blue')
    ax2.set_title('Frame Checker Results')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Detection')
    ax2.set_yticks([0, 1])
    ax2.grid()
    ax2.set_ylim([-0.1, 1.1])
    ax2.set_yticklabels(['Real', 'Adversarial'])
    # plt.show()

    fig3, ax3 = plt.subplots(figsize=figsize)
    ax3.step(range(len(fe_history)), fe_history, where='post', color='red')
    ax3.set_title('Frame Cleaner Activity')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Status')
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['Unused', 'Used'])
    ax3.grid()
    ax3.set_ylim([-0.1, 1.1])
    # plt.show()

    extended_ad_history = np.copy(ad_history)

    for i in range(len(ad_history) - 40):
        if not ad_history[i] and ad_history[i+1]:
            extended_ad_history[i+1:i+41] = False

    fig4, ax4 = plt.subplots(figsize=figsize)
    ax4.set_title('Driving Mode')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Status')
    ax4.step(range(len(extended_ad_history)), extended_ad_history, where='post', color='red', linestyle='dashed', label='ADF Status')
    ax4.step(range(len(ad_history)), ad_history, where='post', color='green', label='DM Status')
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['ADF Available', 'Manual'])
    ax4.legend()
    ax4.grid()
    ax4.set_ylim([-0.1, 1.1])
    
    plt.show()