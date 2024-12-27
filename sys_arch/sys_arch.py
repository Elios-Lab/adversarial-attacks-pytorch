import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/yolo_adv/cleaner')
from yolo_adv.classifier.classifier import Classifier
from ultralytics import YOLO
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
from tqdm import tqdm
import re
from yolo_adv.cleaner.predictor import Predictor as FCL_poltergeist

class FrameSequence:
    def __init__(self, window_size_tor, window_size_ad, frames_path):
        self.window_size_tor = window_size_tor
        self.window_size_ad = window_size_ad
        self.frames = np.array(sorted(os.listdir(frames_path)))
        # np.random.shuffle(self.frames[:350])
        # print('Frames: ', self.frames[:5])
        self.num_of_frames = len(self.frames)
        self.adversarial_detections_tor = np.full(window_size_tor, False)
        self.adversarial_detections_ad = np.full(window_size_ad, False)
        # self.weights = np.linspace(1, 1 / self.window_size, num=self.window_size)
        self.weights_tor = np.exp(-np.linspace(0, 5, num=self.window_size_tor))
        self.weights_ad = np.exp(-np.linspace(0, 5, num=self.window_size_ad))
        self.worst_case_tor = np.sum(self.weights_tor * np.full_like(self.weights_tor, True))
        self.worst_case_ad = np.sum(self.weights_ad * np.full_like(self.weights_ad, True))
        # Initialize the array of attacks to all zeros
        self.attack_indices = np.zeros(len(self.frames))
        self.attack_mapping = {'clean': 0, 'pixle': 1, 'poltergeist': 2}
        # Update the array based on image file names
        for i, frame in enumerate(self.frames):
            for keyword, index in self.attack_mapping.items():
                if keyword.lower().replace("-", "") in frame.lower():
                    self.attack_indices[i] = index
                    break
    
    def sort_key(self,filename):
        num = re.findall(r'\d+', filename)
        return int(num[0]) if num else 0

     
    def update_detections(self, frame_id):
        is_adv = frame_id == 1 or frame_id == 2
        # Add the latest detection and ensure the list size doesn't exceed the window size
        self.adversarial_detections_tor = np.roll(self.adversarial_detections_tor, 1)
        self.adversarial_detections_ad = np.roll(self.adversarial_detections_ad, 1)
        self.adversarial_detections_tor[0] = is_adv
        self.adversarial_detections_ad[0] = is_adv
                
    def calculate_weighted_sum(self, ad=False):
        if not ad:  # if not driving in automated mode, check the AD resume window
            return np.sum(self.weights_ad * self.adversarial_detections_ad) / self.worst_case_ad * 100
        # Apply linearly increasing weights and calculate the weighted sum
        return np.sum(self.weights_tor * self.adversarial_detections_tor) / self.worst_case_tor * 100 # Normalize to 100
    
    def calculate_sum(self, ad=False):
        if not ad:  # if not driving in automated mode, check the AD resume window
            return np.sum(self.adversarial_detections_ad) / self.window_size_ad * 100
        return np.sum(self.adversarial_detections_tor) / self.window_size_tor * 100  # Normalize to 100
    
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

# def temp_jpeg_compression(image, perc=40):
#     # Simulate a JPEG compression that is able to remove adversarial perturbations 40% of the time
#     return np.random.randint(0, 100) > perc
def clean_image(input_path: str, output_path: str, weights_path: str, model_name: str = ''):
    '''
    Clean the image using the DeblurGANv2 model
    input_path: str: Path to the input image
    output_path: str: Path to save the cleaned image (not used)
    weights_path: str: Path to the model weights
    model_name: str: Name of the model architecture
    '''
    predictor = FCL_poltergeist(weights_path, model_name)
    cleaned_img = predictor(image)

    return cleaned_img
        
if __name__ == "__main__":
    seq = 'massive_mix'  # massive
    save_folder = './sys_arch/seq_results/'
    save_folder += seq + '/'
    load_seq_from_file = False
    window_size_tor = 5
    window_size_ad = 200
    to_threshold = 20 #12
    ad_threshold = 0
    frames_path = '/home/lazzaroni/adv/datasets/adv_seq/' + seq + '/'
    if not load_seq_from_file:
        fs = FrameSequence(window_size_tor, window_size_ad, frames_path=frames_path)
        classifier = Classifier('mobilenetv2')
        classifier.load_model(model_path='yolo_adv/classifier/FC_FP16.pt')
        FCL_poltergeist = FCL_poltergeist('yolo_adv/cleaner/fpn_mobilenet.h5')
        tld_model = YOLO('yolo_adv/best.pt')
        todm = DecisionMaker(threshold=to_threshold)
        adm = DecisionMaker(threshold=ad_threshold)
        automated_mode = True  # Start in automated driving mode
        
        fc_post_history = np.full(fs.num_of_frames, 0)  # 'pixle' or 'poltergeist' if frame is classified as adversarial
        fc_pre_history = np.full(fs.num_of_frames, 0)  # 'pixle' or 'poltergeist'  if frame is classified as adv before cleaning
        fe_history = np.full(fs.num_of_frames, False)  # True if frame is enhanced
        ad_history = np.full(fs.num_of_frames, False)  # True while in AD mode
        thrs_history = np.full(fs.num_of_frames, 0, dtype=np.float32)  # Threshold history
       
        for i, frame in tqdm(enumerate(fs.frames), total=fs.num_of_frames, unit='frame'):
            image = Image.open(frames_path+frame)
            # Frame checker
            fc_pre_history[i] = frame_id = fs.attack_mapping[classifier.predict(image=image, amp=True)] 
            # if frame name contains 'PGD', 'F-FGSM', 'VNI-FGSM', 'Pixle', or 'DeepFool', classify as adversarial 85% of the time, else classify it as non-adversarial 85% of the time
            # if any(keyword.lower().replace("-", "") in frame.lower() for keyword in fs.attack_mapping.keys()):
            #     frame_id = np.random.randint(0, 100) <= 84
            # else:
            #     frame_id = np.random.uniform(0, 100) >= 99.75
            # fc_pre_history[i] = frame_id
            if frame_id != 0:  # Frame cleaner
                # image = compress_image_to_jpeg(image, quality=100)
                # frame_id = classifier.predict(image=image) == 'adv'
                # frame_id = temp_jpeg_compression(image, perc=46)
                image = Image.fromarray(FCL_poltergeist(np.array(image), None))
                fc_post_history[i] = frame_id = fs.attack_mapping[classifier.predict(image=image, amp=True)]
                fe_history[i] = True
            # fc_post_history[i] = frame_id
            fs.update_detections(frame_id)
            threshold = fs.calculate_sum(automated_mode)  # fs.calculate_weighted_sum()
            thrs_history[i] = threshold
            if frame_id == 0:  # If the frame is not adversarial, run TLD
                # results = tld_model.predict(image)
                if not automated_mode:  # Driving in manual mode
                    automated_mode = ad_history[i] = adm.check_for_ad(threshold)
                # TODO else output detection results
                else:  # Driving in AD mode and no adv detected, so continue AD
                    automated_mode = ad_history[i] = True
            else: # Frame is adversarial, cleaning not successful
                if not automated_mode: # Driving in manual mode
                    automated_mode = ad_history[i] = adm.check_for_ad(threshold)
                else:  # Driving in AD mode
                    automated_mode = ad_history[i] = not todm.check_for_tor(threshold)  # If TOR is triggered, switch to manual mode
                    
        
        # Convert boolean arrays to integers
        # fc_post_history = fc_post_history.astype(int)
        fe_history = fe_history.astype(int)
        # ad_history = np.logical_not(ad_history).astype(int)
        ad_history = ad_history.astype(int)
    
    figsize = (10, 4)

    # Creating plots
    # Plot attack history attack by attack
    # fig0, ax0 = plt.subplots(figsize=figsize)
    # ax0.step(range(len(fs.attack_indices)), fs.attack_indices, where='post', color='orange')
    # ax0.set_title('Attack History')
    # ax0.set_yticks(list(fs.attack_mapping.values()))
    # ax0.set_yticklabels([key.replace('F-', '') if key.startswith('F-') else key for key in fs.attack_mapping.keys()])
    # ax0.set_xlabel('Step')
    # # ax0.set_ylabel('Attack Type')
    # ax0.grid()
    # # plt.show()
    
    # Plot attack history in binary form (atk-noatk)
    fig0, ax0 = plt.subplots(figsize=figsize)

    # Modify attack_indices to be a binary indicator of whether an attack occurred or not
    name = 'attack_history'
    attack_occurred = np.array([0 if attack == 0 else 1 for attack in fs.attack_indices]) if not load_seq_from_file else np.load(save_folder + name + '.npy')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    np.save(save_folder + 'attack_occurred.npy', attack_occurred)

    ax0.step(range(len(attack_occurred)), attack_occurred, where='post', color='orange')
    ax0.set_title('Attack')
    ax0.set_yticks([0, 1])
    ax0.set_yticklabels(['Not Attacked', 'Attacked'])
    ax0.set_xlabel('Step')
    ax0.grid()
    
    print('Saving into: ' + save_folder)
    np.save(save_folder + name + '.npy', attack_occurred)
    fig0.savefig(save_folder + name + '.svg', transparent=True, pad_inches=0, format='svg')
    
    # Plot threshold history
    fig1, ax1 = plt.subplots(figsize=figsize)
    name = 'threshold_history'
    thrs_history = np.load(save_folder + name + '.npy') if load_seq_from_file else thrs_history
    
    def get_fill_regions(thrs_history, to_threshold, ad_threshold):
        regions = []
        to_threshold_reached = False
        for i in range(len(thrs_history)-1):
            if thrs_history[i] > to_threshold and not to_threshold_reached:
                to_threshold_reached = True
            elif thrs_history[i] <= ad_threshold and to_threshold_reached:
                to_threshold_reached = False
            regions.append(to_threshold_reached)
        return regions
    
    fill_regions = get_fill_regions(thrs_history, to_threshold, ad_threshold)

    for i in range(len(thrs_history)-1):
        color = 'darkred' if fill_regions[i] else 'darkgreen'
        ax1.plot([i, i+1], [thrs_history[i], thrs_history[i+1]], color=color)
    ax1.set_title(r'$\widetilde{A}$')
    ax1.set_xlabel('Step')
    ax1.grid()
    ax1.set_ylim([-5, 45])
    vals = ax1.get_yticks()
    ax1.set_yticklabels(['{:,.0f}%'.format(x) for x in vals])
    ax1.axhline(y=to_threshold, linestyle='--', color='red')
    ax1.axhline(y=ad_threshold, linestyle='--', color='lawngreen')
    for i in range(len(fill_regions)):
        if fill_regions[i]:
            ax1.fill_between([i, i+1], [ad_threshold, ad_threshold], [to_threshold, to_threshold], color='pink')
        else:
            ax1.fill_between([i, i+1], [ad_threshold, ad_threshold], [to_threshold, to_threshold], color='lightgreen')
    print('Saving into: ' + save_folder)
    np.save(save_folder + name + '.npy', thrs_history)
    fig1.savefig(save_folder + name + '.svg', transparent=True, pad_inches=0, format='svg')
    
    
    # Plot Frame Checker history
    fig2, ax2 = plt.subplots(figsize=figsize)
    
    name = 'frame_checker'
    fc_pre_history = np.load(save_folder + name + 'pre.npy') if load_seq_from_file else fc_pre_history
    fc_post_history = np.load(save_folder + name + 'post.npy') if load_seq_from_file else fc_post_history
    # ax2.step(range(len(fc_post_history)), fc_post_history, where='post', color='red', label='Attack Confirmed')
    # ax2.step(range(len(fc_pre_history)), fc_pre_history, where='post', color='blue', label='Frame Cleaner')
    # Create a new array that is True only when both fc_pre_history and fc_post_history are True
    # both_attacked = np.logical_and(fc_pre_history, fc_post_history)
    ax2.step(range(len(fc_pre_history)), fc_pre_history, where='post', color='blue', label='Pre-cleaning')
    ax2.step(range(len(fc_post_history)), fc_post_history, where='post', color='red', label='Post-cleaning')

    ax2.set_title('Frame Checker')
    ax2.set_xlabel('Step')
    # ax2.set_ylabel('Detection')
    ax2.set_yticks(list(fs.attack_mapping.values()))
    ax2.grid()
    ax2.set_yticklabels([key.replace('F-', '') if key.startswith('F-') else key for key in fs.attack_mapping.keys()])
    ax2.legend()
    
    print('Saving into: ' + save_folder)
    np.save(save_folder + name + 'pre.npy', fc_pre_history)
    np.save(save_folder + name + 'post.npy', fc_post_history)
    fig2.savefig(save_folder + name + '.svg', transparent=True, pad_inches=0, format='svg')
    # plt.show()

    # fig3, ax3 = plt.subplots(figsize=figsize)
    # ax3.step(range(len(fe_history)), fe_history, where='post', color='blue')
    # ax3.set_title('Frame Cleaner Activity')
    # ax3.set_xlabel('Step')
    # ax3.set_ylabel('Status')
    # ax3.set_yticks([0, 1])
    # ax3.set_yticklabels(['Unused', 'Used'])
    # ax3.grid()
    # ax3.set_ylim([-0.1, 1.1])
    # plt.show()
    
    # Plot Automated Driving Mode history
    fig4, ax4 = plt.subplots(figsize=figsize)
    
    name = 'adf_state'
    ad_history = np.load(save_folder + name + '.npy') if load_seq_from_file else ad_history
    extended_ad_history = np.copy(ad_history)    
    for i in range(len(ad_history) - 40):
        if ad_history[i] and not ad_history[i+1]:
            extended_ad_history[i+1:i+41] = True
            ax4.axvline(x=i+1, linestyle='--', color='red', label='TOR')
    ax4.set_title('ADF')
    ax4.set_xlabel('Step')
    # ax4.set_ylabel('State')
    ax4.step(range(len(extended_ad_history)), extended_ad_history, where='post', color='green', label='ADF State')
    # ax4.step(range(len(ad_history)), ad_history, where='post', color='green', label='DM Status')
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['Not available', 'Available'])
    ax4.legend()
    ax4.grid()
    ax4.set_ylim([-0.1, 1.1])
    
    print('Saving into: ' + save_folder)
    np.save(save_folder + name + '.npy', ad_history)
    fig4.savefig(save_folder + name + '.svg', transparent=True, pad_inches=0, format='svg')
    
    plt.show()