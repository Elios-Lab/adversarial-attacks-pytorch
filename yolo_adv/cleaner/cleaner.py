import os
import cv2
import numpy as np
import torch
from predictor import Predictor

def clean_image(input_path: str, output_path: str, weights_path: str, model_name: str = ''):
    '''
    Clean the image using the DeblurGANv2 model
    input_path: str: Path to the input image
    output_path: str: Path to save the cleaned image (not used)
    weights_path: str: Path to the model weights
    model_name: str: Name of the model architecture
    '''
    img = cv2.imread(input_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predictor = Predictor(weights_path, model_name)
    cleaned_img = predictor(img)
    cleaned_img = cv2.cvtColor(cleaned_img, cv2.COLOR_RGB2BGR)

    return cleaned_img