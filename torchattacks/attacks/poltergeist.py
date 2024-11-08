import math
import numpy as np
from tqdm import tqdm
import os
import cv2
from numba import jit

@jit(nopython=True)
def cal_blur(imgarray, theta, delta, L, S=0):
    imgheight = imgarray.shape[0]
    imgwidth = imgarray.shape[1]
    c0 = int(imgheight / 2)
    c1 = int(imgwidth / 2)
    theta = theta / 180 * math.pi
    delta = delta / 180 * math.pi
    blurred_imgarray = np.empty_like(imgarray, dtype=np.float32)
    
    for x in range(imgheight):
        for y in range(imgwidth):
            R = math.sqrt((x - c0) ** 2 + (y - c1) ** 2)
            alpha = math.atan2(y - c1, x - c0)
            X_cos = L * math.cos(delta) - S * R * math.cos(alpha)
            Y_sin = L * math.sin(delta) - S * R * math.sin(alpha)
            N = max(int(abs(R * math.cos(alpha + theta) + X_cos + c0 - x)),
                    int(abs(R * math.sin(alpha + theta) + Y_sin + c1 - y)))
            
            if N <= 1:
                blurred_imgarray[x, y] = imgarray[x, y]
                continue
            
            count = 0
            sum_r, sum_g, sum_b = 0.0, 0.0, 0.0
            for i in range(N + 1):
                n = i / N
                xt = int(R * math.cos(alpha + n * theta) + n * X_cos + c0)
                yt = int(R * math.sin(alpha + n * theta) + n * Y_sin + c1)
                
                if 0 <= xt < imgheight and 0 <= yt < imgwidth:
                    sum_r += imgarray[xt, yt][0]
                    sum_g += imgarray[xt, yt][1]
                    sum_b += imgarray[xt, yt][2]
                    count += 1
            
            if count > 0:
                blurred_imgarray[x, y] = np.array([sum_r / count, sum_g / count, sum_b / count])
            else:
                blurred_imgarray[x, y] = imgarray[x, y]
    
    blurred_imgarray = np.clip(blurred_imgarray, 0, 255).astype(np.uint8)
    return blurred_imgarray

source_dir = r"C:\Users\elios\Downloads\BoschDriveU5px\valid\images"
target_dir = r"C:\Users\elios\Downloads\BoschDriveU5px\polter_valid"

if not os.path.exists(target_dir):
    os.makedirs(target_dir)
thetas = [0.5, 1, 2, 5]

for i, filename in tqdm(enumerate(os.listdir(source_dir)), total=len(os.listdir(source_dir)), desc="Applying Poltergeist Attack"):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(source_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        theta = thetas[i % len(thetas)]
        blurred_img = cal_blur(img, theta, 0, 0, 0)
        new_filename = f"{os.path.splitext(filename)[0]}_theta_{theta}.jpg"
        cv2.imwrite(os.path.join(target_dir, new_filename), blurred_img)