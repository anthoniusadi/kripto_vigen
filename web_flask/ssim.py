
import math
import numpy as np
import cv2
import modules
import matplotlib.pyplot as plt

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
    
    sigma1_sq = cv2.filter2D(img1**2, -1, window)- mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window) - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window) - mu1_mu2
    
    atas = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
    bawah = ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim_map = atas / bawah
    cv2.imshow('im',ssim_map)
    cv2.waitKey()
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
    
img1=cv2.imread('/home/epiphany/FAZ project/kripto_vigen/web_flask/static/output/image_dekripsi.jpg')
img2=cv2.imread('/home/epiphany/FAZ project/kripto_vigen/web_flask/static/output/original.jpg')
gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)   
gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)   

value = calculate_ssim(img1,img2)
# value = modules.ssim_evaluation('/home/epiphany/FAZ project/kripto_vigen/web_flask/static/output/image_dekripsi.jpg','/home/epiphany/FAZ project/kripto_vigen/web_flask/static/output/original.jpg')

print(value)