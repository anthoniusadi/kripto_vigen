from math import sqrt
import os
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

def entropy(img1):
        val_entropy = []
        img = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)  

        hist = cv2.calcHist([img], [0], None, [256], [0, 255])
        total_pixel = img.shape[0] * img.shape[1]

        for item in hist:
            probability = item / total_pixel
            if probability == 0:
                en = 0
            else:
                en = -1 * probability * (np.log(probability) / np.log(2))
            val_entropy.append(en)

        sum_en = np.sum(val_entropy)
        return sum_en[0]

def calc_mse(img1,img2):
    error_pixel = (img1 -img2) ** 2
    summed_error = np.sum(error_pixel)
    total_pixel = img1.shape[0] * img1.shape[1] 
    mse_val = summed_error / total_pixel
    return mse_val

def calc_psnr(img1,img2):
#     img1 = cv2.imread(img1)
#     img2 = cv2.imread(img2)
#     calc PSNR
    # mse = calc_mse(img1,img2)
    # print(mse)
    # mse = np.mean((img1.astype(np.float64) / 255 - img2.astype(np.float64) / 255) ** 2)
    # psnr = 10 * np.log10(255.0 / mse)
    # psnr = 20 * np.log10(1.0 / sqrt(mse)) 
    psnr =  cv2.PSNR(img1,img2)
    return psnr

def D(source_img,restored_img):
    b_channel_source,g_channel_source,r_channel_source = cv2.split(source_img)
    b_channel_restore,g_channel_restore,r_channel_restore = cv2.split(restored_img)
    count_0 =0
    count_1 =0
    for i in range (b_channel_restore.shape[0]):
        for j in range(b_channel_restore.shape[1]):
            if b_channel_source[i][j] == b_channel_restore[i][j]:
                count_0+=1
            else:
                count_1+=1
    for i in range (g_channel_restore.shape[0]):
        for j in range(g_channel_restore.shape[1]):
            if g_channel_source[i][j] == g_channel_restore[i][j]:
                count_0+=1
            else:
                count_1+=1
    for i in range (b_channel_restore.shape[0]):
        for j in range(b_channel_restore.shape[1]):
            if r_channel_source[i][j] == r_channel_restore[i][j]:
                count_0+=1
            else:
                count_1+=1
    return count_0,count_1

def npcr(img1,img2):
    _ , one = D(img1,img2)
    return one /(img1.shape[0]*img1.shape[1])

def uaci(source_img,restored_img):
    b_channel_source,g_channel_source,r_channel_source = cv2.split(source_img)
    b_channel_restore,g_channel_restore,r_channel_restore = cv2.split(restored_img)
    b = np.sum(abs(b_channel_source-b_channel_restore))
    g = np.sum(abs(g_channel_source-g_channel_restore))
    r = np.sum(abs(r_channel_source-r_channel_restore))  
    s=(b+g+r)/255.0
#     print(s)
    value = round((s / (source_img.shape[0]*source_img.shape[1]) )*100,2)
#     print(value)
    return value