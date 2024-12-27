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
    
    frames_path = '/Users/luca/Documents/adv/datasets/polter_FINAL/polter_strong_01_20'
    frames = sorted(os.listdir(frames_path))
    attack_mapping = {'clean': 0, 'pixle': 1, 'poltergeist': 2}
    classifier = Classifier('mobilenetv2')
    classifier.load_model(model_path='yolo_adv/classifier/FC_FP16.pt')
    FCL_poltergeist = FCL_poltergeist('yolo_adv/cleaner/fpn_mobilenet.h5')
    tld_model = YOLO('yolo_adv/best.pt')
    
    for i, frame in tqdm(enumerate(frames[:3]), total=len(frames), unit='frame'):
        image = Image.open(frames_path+frame)
        # clean
        image = Image.fromarray(FCL_poltergeist(np.array(image), None))
        results = tld_model.predict(image)
        print(results)