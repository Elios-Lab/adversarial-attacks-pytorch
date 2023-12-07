from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO('yolo_adv/best.pt')

# img = cv2.imread('filtered/PGD_0.png')
# cv2.imwrite('filtered/PGD_0.jpg', img)
       
results = model('/home/pigo/projects/adversarial-attacks-pytorch/yolo_adv/adv_data/norm/images/dayClip7--00930.jpg', verbose=False)
print(results[0].boxes.conf)
print(results[0].boxes.cls)
# for r in results:
#     im_array = r.plot()
#     im = Image.fromarray(im_array[..., ::-1])
#     im.show()