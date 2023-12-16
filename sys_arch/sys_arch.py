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
        self.current_index = 0
        self.weights = np.linspace(1, 1 / self.window_size, num=self.window_size)
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
        self.adversarial_detections[self.current_index] = is_adversarial
        self.current_index = (self.current_index + 1) % self.window_size
                
    def calculate_weighted_sum(self):
        # Apply linearly increasing weights and calculate the weighted sum
        weighted_sum = np.sum(self.weights * self.adversarial_detections)
        return weighted_sum
    
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
        if ws < self.threshold:
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
        
if __name__ == "__main__":
    window_size = 200
    frames_path = '/Users/luca/Documents/adv_sequences_atk/seq2/images/'
    fs = FrameSequence(window_size, frames_path=frames_path)
    classifier = Classifier()
    classifier.load_model(model_path='yolo_adv/classifier/last.pt')
    tld_model = YOLO('yolo_adv/best.pt')
    todm = DecisionMaker(threshold=10)
    adm = DecisionMaker(threshold=2)
    automated_mode = True  # Start in automated driving mode
    
    fc_history = np.full(fs.num_of_frames, False)  # True if frame is classified as clean
    fe_history = np.full(fs.num_of_frames, False)  # True if frame is enhanced
    ad_history = np.full(fs.num_of_frames, False)  # True while in AD mode
    thrs_history = np.full(fs.num_of_frames, 0)  # Threshold history
    
    for i, frame in enumerate(fs.frames):
        image = Image.open(frames_path+frame)
        is_adversarial = classifier.predict(image=image) == 'adv'  
        if is_adversarial:  # Try frame enhancement
            image = compress_image_to_jpeg(image, quality=100)
            is_adversarial = classifier.predict(image=image) == 'adv'
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
    
    figsize = (7.0625, 3)

    # Creating plots
    fig, ax = plt.subplots(figsize=figsize)
    ax.step(range(len(fs.attack_indices)), fs.attack_indices, where='post', color='orange')
    ax.set_yticks(list(fs.attack_mapping.values()))
    ax.set_yticklabels(list(fs.attack_mapping.keys()))
    ax.set_xlabel('Step')
    ax.set_ylabel('Attack Type')
    ax.grid()
    plt.show()
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.step(range(len(thrs_history)), thrs_history, where='post', color='pink')
    ax.set_title('Threshold History')
    ax.set_xlabel('Step')
    ax.set_ylabel('Value')
    ax.grid()
    plt.show()
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.step(range(len(fc_history)), fc_history, where='post', color='blue')
    ax.set_title('Frame Checker Results')
    ax.set_xlabel('Step')
    ax.set_ylabel('Detection')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Real', 'Adversarial'])
    ax.grid()
    ax.set_ylim([-0.1, 1.1])
    plt.show()

    fig, ax = plt.subplots(figsize=figsize)
    ax.step(range(len(fe_history)), fe_history, where='post', color='red')
    ax.set_title('Frame Enhancer Activity')
    ax.set_xlabel('Step')
    ax.set_ylabel('Status')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Disabled', 'Enabled'])
    ax.grid()
    ax.set_ylim([-0.1, 1.1])
    plt.show()

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title('Driving Mode')
    ax.set_xlabel('Step')
    ax.set_ylabel('Status')
    ax.step(range(len(ad_history)), ad_history, where='post', color='green')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Automated', 'Manual'])
    ax.grid()
    ax.set_ylim([-0.1, 1.1])
    plt.show()