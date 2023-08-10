from math import sqrt
import os
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import gauss
from scipy import signal
from scipy import ndimage
def ssim_evaluation(img1, img2, cs_map=False):

    img1 = np.asarray(Image.open(img1).convert('L'))
    img2 = np.asarray(Image.open(img2).convert('L'))
    print(img1.shape)
    print(img2.shape)
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    
    score = ssim(img1,img2,data_range=1.0)
    return score
    
    # size = 11
    # sigma = 1.5
    # window = np.random.normal(size, sigma)
    # K1 = 0.01
    # K2 = 0.03
    # L = 255 #bitdepth of image
    # C1 = (K1*L)**2
    # C2 = (K2*L)**2
    # mu1 = signal.fftconvolve(window, img1, mode='valid')
    # mu2 = signal.fftconvolve(window, img2, mode='valid')
    # mu1_sq = mu1*mu1
    # mu2_sq = mu2*mu2
    # mu1_mu2 = mu1*mu2
    # sigma1_sq = signal.fftconvolve(window, img1*img1, mode='valid') - mu1_sq
    # sigma2_sq = signal.fftconvolve(window, img2*img2, mode='valid') - mu2_sq
    # sigma12 = signal.fftconvolve(window, img1*img2, mode='valid') - mu1_mu2
    # if cs_map:
    #     return (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
    #                 (sigma1_sq + sigma2_sq + C2)), 
    #             (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    # else:
    #     return ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
    #                 (sigma1_sq + sigma2_sq + C2))
def hitung_ssim(img1, img2):
    
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    # window dibuat 2dimensi gausian
    window = np.outer(kernel, kernel.transpose())
# -1 supaya dimensi outputnya sama dengan input 
    mu1 = cv2.filter2D(img1, -1, window)
    mu2 = cv2.filter2D(img2, -1, window)
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.filter2D(img1**2, -1, window) - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window) - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window) - mu1_mu2
    
    atas = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
    bawah = ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim_map = atas / bawah
    # cv2.imshow('im',ssim_map)
    # cv2.waitKey()
    # plt.imshow((ssim_map*255).astype(np.uint8))
    # plt.colorbar()
    # plt.show()
    print(f'window : {window}, kernel : {kernel}')
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    
    if not img1.shape == img2.shape:
        raise ValueError('Dimensi berbeda')
    if img1.ndim == 2:
        return hitung_ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(hitung_ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return hitung_ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Dimensi tidak sama')
    
def ncc_evaluation(img1, img2):
    atas = np.mean((img1 - img1.mean()) * (img2 - img2.mean()))
    standard_dev = img1.std() * img2.std()
    if standard_dev == 0:
        return 0
    else:
        atas /= standard_dev
        value = atas
        return value
    
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
        return round(sum_en[0],3)

def calc_mse(img1,img2):
    if (img1.shape[0] == img2.shape[0] and img1.shape[1] == img2.shape[1]):
        error_pixel = (img1 -img2) ** 2
        summed_error = np.sum(error_pixel)
        total_pixel = img1.shape[0] * img1.shape[1] 
        mse_val = summed_error / total_pixel
        return round(mse_val,3)
    else:
        return 'shape doesnt match'

# def calc_mse(img1,img2):
#     if (img1.shape[0] == img2.shape[0] and img1.shape[1] == img2.shape[1]):
#         mse = np.mean(((img1.astype(np.float64) / 255 )- (img2.astype(np.float64) / 255)) ** 2)
#         return round(mse,3)
#     else:
#         return 'shape doesnt match'

def calc_psnr(img1,img2):
#     img1 = cv2.imread(img1)
#     img2 = cv2.imread(img2)
#     calc PSNR
    # mse = calc_mse(img1,img2)
    # print(mse)
    if (img1.shape[0] == img2.shape[0] and img1.shape[1] == img2.shape[1]):
        mse = np.mean((img1.astype(np.float64) / 255 - img2.astype(np.float64) / 255) ** 2)
    # psnr = 10 * np.log10(255.0 / mse)
        if mse>0:
            psnr = 20 * np.log10(1.0 / sqrt(mse)) 
            return round(psnr,3)
        else:
            psnr = 'INFINITY'
        # psnr =  cv2.PSNR(img1,img2)
            return psnr
    else:
        return "shape doesnt match"

def D(source_img,restored_img):
    if (source_img.shape[0] == restored_img.shape[0] and source_img.shape[1] == restored_img.shape[1]):
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
    else:
        return "shape doesnt match"

def npcr(img1,img2):
    if (img1.shape[0] == img2.shape[0] and img1.shape[1] == img2.shape[1]):
        _ , one = D(img1,img2)
        return round(one /(img1.shape[0]*img1.shape[1]),3)
    else:
        return "shape doesnt match"

def uaci(source_img,restored_img):
    if (source_img.shape[0] == restored_img.shape[0] and source_img.shape[1] == restored_img.shape[1]):

        b_channel_source,g_channel_source,r_channel_source = cv2.split(source_img)
        b_channel_restore,g_channel_restore,r_channel_restore = cv2.split(restored_img)
        b = np.sum(abs(b_channel_source-b_channel_restore))
        g = np.sum(abs(g_channel_source-g_channel_restore))
        r = np.sum(abs(r_channel_source-r_channel_restore))  
        s=(b+g+r)/255.0
    #     print(s)
        value = round((s / (source_img.shape[0]*source_img.shape[1]) )*100,2)
    #     print(value)
        return round(value,3)
    else:
        return "shape doesnt match"