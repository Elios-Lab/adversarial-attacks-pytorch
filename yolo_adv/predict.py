from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO('yolo_adv/best.pt')

results = model.val(data='yolo_adv/data.yaml', imgsz=640)
print('mAP50-95:', results.box.map)    # map50-95
print('mAP50:', results.box.map50)  # map50
print('mAP75:', results.box.map75)  # map75
print('mAP for each class:', results.box.maps)  # a list contains map50-95 of each category
print('Mean precision:', results.box.mp)  # mean precision
print('Mean recall:', results.box.mr)