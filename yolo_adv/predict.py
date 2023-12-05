from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO('yolo_adv/best.pt')

# img = cv2.imread('filtered/PGD_0.png')
# cv2.imwrite('filtered/PGD_0.jpg', img)
       
results = model('filtered/VNIFGSM_0.png', verbose=False)

for r in results:
    im_array = r.plot()
    im = Image.fromarray(im_array[..., ::-1])
    im.show()