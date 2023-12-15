from yolo_adv.classifier import Classifier
from ultralytics import YOLO
import os
import numpy as np
import matplotlib.pyplot as plt

class FrameSequence:
    def __init__(self, window_size, frames_path):
        self.window_size = window_size
        self.frames = os.listdir(frames_path)
        self.num_of_frames = len(self.frames)
        self.adversarial_detections = np.full(window_size, False)
        self.current_index = 0
        # Initialize the array of attacks to all zeros
        self.attack_indices = np.zeros(len(self.frames))
        self.attack_mapping = {'None': 0, 'PGD': 1, 'FGSM': 2, 'F-FGSM': 3, 'VNI-FGSM': 4, 'Pixle': 5, 'DeepFool': 6}
        # Update the array based on image file names
        for i, frame in enumerate(self.frames):
            for keyword, index in self.attack_mapping.items():
                if keyword.lower() in frame.lower():
                    self.attack_indices[i] = index
                    break
     
    def update_detections(self, is_adversarial):
        # Add the latest detection and ensure the list size doesn't exceed the window size
        self.adversarial_detections[self.current_index] = is_adversarial
        self.current_index = (self.current_index + 1) % self.window_size
                
    def calculate_weighted_sum(self):
        # Apply linearly increasing weights and calculate the weighted sum
        weights = np.linspace(1 / self.window_size, 1, num=self.window_size)
        weighted_sum = np.sum(weights * self.adversarial_detections)
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
        
if __name__ == "__main__":
    window_size = 200
    fs = FrameSequence(window_size, frames_path='path/to/frames')
    classifier = Classifier()
    classifier.load_model(model_path='path/to/classifier/model/.pt')
    tld_model = YOLO("path/to/yolo/model/.pt")
    todm = DecisionMaker(threshold=10)
    adm = DecisionMaker(threshold=8)
    automated_mode = True  # Start in automated driving mode
    
    fc_history = np.full(fs.num_of_frames, False)  # True if frame is classified as clean
    fe_history = np.full(fs.num_of_frames, False)  # True if frame is enhanced
    ad_history = np.full(fs.num_of_frames, False)  # True while in AD mode
    
    for i, frame in enumerate(fs.frames):
        is_adversarial = classifier.predict(image_path=frame) == 'adv'  
        if is_adversarial:  # Try frame enhancement
            frame_enhanced = ...
            frame = frame_enhanced
            is_adversarial = classifier.predict(image_path=frame) == 'adv'
            fe_history[i] = True
        fc_history[i] = is_adversarial
        fs.update_detections(is_adversarial)
        threshold = fs.calculate_weighted_sum()
        if not is_adversarial:  # If the frame is not adversarial, run TLD
            results = tld_model.predict(frame)
            if not automated_mode:  # Driving in AD mode
                automated_mode = ad_history[i] = adm.check_for_ad(threshold)
            # TODO else output detection results
        else: # Frame is adversarial, enhancement not successful
            if not automated_mode: # Driving in manual mode
                automated_mode = ad_history[i] = adm.check_for_ad(threshold)
            else:  # Driving in AD mode
                automated_mode = ad_history[i] = not todm.check_for_tor()  # If TOR is triggered, switch to manual mode
    
    # Convert boolean arrays to integers
    fc_history = fc_history.astype(int)
    fe_history = fe_history.astype(int)
    ad_history = np.logical_not(ad_history).astype(int)

    # Creating plots
    fig, ax = plt.subplots(figsize=(7.5, 3))
    ax.step(range(len(fs.attack_indices)), fs.attack_indices, where='post', color='orange')
    ax.set_yticks(list(fs.attack_mapping.values()))
    ax.set_yticklabels(list(fs.attack_mapping.keys()))
    ax.set_xlabel('Step')
    ax.set_ylabel('Attack Type')
    ax.grid()
    plt.show()
    
    
    fig, ax = plt.subplots(figsize=(7.5, 3))
    ax.step(range(len(fc_history)), fc_history, where='post', color='blue')
    ax.set_title('Frame Checker Results')
    ax.set_xlabel('Step')
    ax.set_ylabel('Detection')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Real', 'Adversarial'])
    ax.grid()
    ax.set_ylim([-0.1, 1.1])
    plt.show()

    fig, ax = plt.subplots(figsize=(7.5, 3))
    ax.step(range(len(fe_history)), fe_history, where='post', color='red')
    ax.set_title('Frame Enhancer Activity')
    ax.set_xlabel('Step')
    ax.set_ylabel('Status')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Disabled', 'Enabled'])
    ax.grid()
    ax.set_ylim([-0.1, 1.1])
    plt.show()

    fig, ax = plt.subplots(figsize=(7.5, 3))
    ax.set_title('Driving Mode')
    ax.set_xlabel('Step')
    ax.set_ylabel('Status')
    ax.step(range(len(ad_history)), ad_history, where='post', color='green')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Automated', 'Manual'])
    ax.grid()
    ax.set_ylim([-0.1, 1.1])
    plt.show()