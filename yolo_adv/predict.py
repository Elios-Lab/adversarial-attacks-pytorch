from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt

model = YOLO('/home/pigo/adversarial-attacks-pytorch/yolo_adv/best.pt')

results = model.val(data='/home/pigo/adversarial-attacks-pytorch/yolo_adv/data.yaml')  # Replace with the path to your dataset

print('mAP50-95:', round(results.box.map,2))    # map50-95
print('mAP50:', round(results.box.map50,2))  # map50
print('mAP75:', round(results.box.map75,2))  # map75
print('mAP for each class:', results.box.maps)  # a list contains map50-95 of each category
print('Mean precision:', round(results.box.mp,2))  # mean precision
print('Mean recall:', round(results.box.mr,2))
f1_score = 2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr)
print('Mean F1 score:', round(f1_score,2))