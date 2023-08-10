from glob import glob
import numpy as np
import os
import modules
from modules.cleaning import rename
import cv2
import math
from skimage import io, feature
from scipy import ndimage

from matplotlib import pyplot as plt

window_size_width = 7
window_size_height = 7
lineThickness = 2

def correlation_coefficient(patch1, patch2):
    product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
    stds = patch1.std() * patch2.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return product
img1 = cv2.imread('/home/epiphany/FAZ project/kripto_vigen/web_flask/static/output/image_dekripsi.jpg')
img2 = cv2.imread('/home/epiphany/FAZ project/kripto_vigen/web_flask/static/output/original.jpg')
width, height, ch = img1.shape[::]
gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)   
gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)   
img2_copy = img2.copy()


corners1 = cv2.goodFeaturesToTrack(gray1, 30, 0.01, 5)
corners1 = np.int0(corners1)

corners2 = cv2.goodFeaturesToTrack(gray2, 30, 0.01, 5)
corners2 = np.int0(corners2)

corners_windows1 = []

for i in corners1:
    x, y = i.ravel()
    cv2.circle(img1, (x, y), 3, 255, -1)

corners_windows2 = []
for i in corners2:
    x, y = i.ravel()
    cv2.circle(img2, (x, y), 3, 255, -1)

plt.imshow(img1), plt.show()

methods = ['NCC']
for method in methods:
    matches = []
    for id1, i in enumerate(corners1):
        x1, y1 = i.ravel()
        if y1 - window_size_height < 0 or y1 + window_size_height > height or x1 - window_size_width < 0 or x1 + window_size_width > width:
            continue
        pt1 = (x1, y1)
      
        template = img1[y1 - window_size_height:y1 + window_size_height, x1 - window_size_width:x1 + window_size_width]
        max_val = 0
        Threshold = 1000000
        id_max = 0
        for id2, i in enumerate(corners2):
            x2, y2 = i.ravel()

            if y2 - window_size_height < 0 or y2 + window_size_height > height or x2 - window_size_width < 0 or x2 + window_size_width > width:
                continue
            window2 = img2[y2 - window_size_height:y2 + window_size_height,
                      x2 - window_size_width:x2 + window_size_width]
            if method == 'SSD':
                temp_min_val = np.sum((template - window2) ** 2)
            elif method == 'NCC':
                temp_min_val = correlation_coefficient(template, window2)
            if temp_min_val < Threshold:
                Threshold = temp_min_val
                pt2 = (x2 + 663, y2)
        matches.append((pt1, pt2))
    stacked_img = np.hstack((img1, img2))
    #show the first 15 matches
    # for match in matches[:15]:
    #     cv2.line(stacked_img, match[0], match[1], (0, 255, 0), lineThickness)
    # matches = []
    plt.imshow(stacked_img), plt.show()
    print(temp_min_val)