import cv2
import os
from ultralytics import YOLO
from PIL import Image

img = cv2.imread('/home/pigo/adversarial-attacks-pytorch/yolo_adv/adv_data/adv/adv_img/PGD/images/20210616_115220_cam_0_00005068_PGD.png')

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
result, encimg = cv2.imencode('.jpg', img, encode_param)


#decode from jpeg format
decimg=cv2.imdecode(encimg,1)

# cv2.imshow('Source Image',img)
# cv2.imshow('Decoded image',decimg)
# cv2.waitKey(0)
# cv2.destroyAllWindows() 
# print(encimg)

model = YOLO('/home/pigo/adversarial-attacks-pytorch/yolo_adv/best.pt')

results = model(decimg, verbose=False)

for r in results:
    im_array = r.plot()
    im = Image.fromarray(im_array[..., ::-1])
    im.show()