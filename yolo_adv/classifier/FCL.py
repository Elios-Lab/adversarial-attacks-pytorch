import cv2
import os
from ultralytics import YOLO
from PIL import Image
from tqdm import tqdm
from torchvision import transforms


jpeg_quality = 95  # A value between 0 and 100 (higher means better quality, but larger file size)
ds_path = r'C:\Users\luca\Documents\FC_test\real'
new_path = r'C:\Users\luca\Documents\FC_post_FCL_test'

for pic in os.listdir(ds_path):
    img = Image.open(rf'{ds_path}\{pic}').convert('RGB')
    resize_transform = transforms.Resize((480, 640))
    img = resize_transform(img)
    
    img.save(rf'{new_path}\real\{pic}.jpg', format='JPEG', quality=jpeg_quality)