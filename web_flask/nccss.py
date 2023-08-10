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

cv2.imshow('image',gray1)
cv2.waitKey()

temp_min_val = correlation_coefficient(img1, img2)
print(temp_min_val)

