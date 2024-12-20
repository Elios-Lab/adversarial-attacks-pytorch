from PIL import Image
import os
import sys
sys.path.append('/home/lazzaroni/adv/adversarial-attacks-pytorch')
sys.path.append('/home/lazzaroni/adv/adversarial-attacks-pytorch/yolo_adv/cleaner')

from yolo_adv.cleaner.predictor import Predictor as FCL_poltergeist
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    image = Image.open('/home/lazzaroni/adv/datasets/giovanni.png')
    FCL = FCL_poltergeist('yolo_adv/cleaner/fpn_mobilenet.h5')
    image_clean = FCL(np.array(image), None)
    print(type(image_clean))
    print(np.max(image_clean))
    print(np.shape(image_clean))
    img = Image.fromarray(image_clean)
    # show the image with matplotlib
    plt.imshow(img)
    plt.show()
    