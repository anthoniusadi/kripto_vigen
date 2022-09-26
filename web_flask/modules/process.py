
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time

# pip install pycryptodome

def __encDecImage(key, pixels, isEncrypt):
    ans = []
    if not isinstance(key, str):
        raise Exception("Key must be string")
    row, col = pixels.shape
    pixels = pixels.flatten()
    
    for i in range(len(pixels)):
        pixel = pixels[i]
        keychar = key[i % len(key)]
        alphIndex = pixel
        alphIndex += isEncrypt * ord(keychar)
        alphIndex %= 256
        ans.append(alphIndex)
        
    ret = np.array(ans).reshape([row, col])
    return np.uint8(ret)

def encryptImage(pixels, key):
    return __encDecImage(key, pixels, 1)

def decryptImage(pixels, key):
    return __encDecImage(key, pixels, -1)

def histogram(source_img):

    img = cv2.imread(source_img)
    b_channel ,g_channel, r_channel = cv2.split(img)
    plt.figure(figsize=(12, 4))
    plt.subplot(131),plt.hist(r_channel.flatten(), bins='auto'),plt.title('R')
    plt.subplot(132),plt.hist(g_channel.flatten(), bins='auto'),plt.title('G')
    plt.subplot(133),plt.hist(b_channel.flatten(), bins='auto'),plt.title('B')
    plt.tight_layout()
    plt.savefig('static/output/image_source_histogram.png')

    plt.figure(figsize=(12, 8))
    plt.subplot(131),plt.imshow(r_channel,'gray'),plt.title('R')
    plt.subplot(132),plt.imshow(g_channel,'gray'),plt.title('G')
    plt.subplot(133),plt.imshow(b_channel,'gray'),plt.title('B')
    plt.tight_layout()
    plt.savefig('static/output/image_source_split.png')
    return b_channel,g_channel,r_channel

def histogram_non(im1,im2):
    img1 = cv2.imread(im1)
    img2 = cv2.imread(im2)
    b_channel ,g_channel, r_channel = cv2.split(img1)
    plt.figure(figsize=(12, 4))
    plt.subplot(131),plt.hist(r_channel.flatten(), bins='auto'),plt.title('R')
    plt.subplot(132),plt.hist(g_channel.flatten(), bins='auto'),plt.title('G')
    plt.subplot(133),plt.hist(b_channel.flatten(), bins='auto'),plt.title('B')
    plt.tight_layout()
    plt.savefig('static/UPLOAD_FOLDER_WO_VIGENERE/citra_uji1_histogram.png')

    plt.figure(figsize=(12, 8))
    plt.subplot(131),plt.imshow(r_channel,'gray'),plt.title('R')
    plt.subplot(132),plt.imshow(g_channel,'gray'),plt.title('G')
    plt.subplot(133),plt.imshow(b_channel,'gray'),plt.title('B')
    plt.tight_layout()
    plt.savefig('static/UPLOAD_FOLDER_WO_VIGENERE/citra_uji1_split.png')

    
    b_channel ,g_channel, r_channel = cv2.split(img2)
    plt.figure(figsize=(12, 4))
    plt.subplot(131),plt.hist(r_channel.flatten(), bins='auto'),plt.title('R')
    plt.subplot(132),plt.hist(g_channel.flatten(), bins='auto'),plt.title('G')
    plt.subplot(133),plt.hist(b_channel.flatten(), bins='auto'),plt.title('B')
    plt.tight_layout()
    plt.savefig('static/UPLOAD_FOLDER_WO_VIGENERE/citra_uji2_histogram.png')


    plt.figure(figsize=(12, 8))
    plt.subplot(131),plt.imshow(r_channel,'gray'),plt.title('R')
    plt.subplot(132),plt.imshow(g_channel,'gray'),plt.title('G')
    plt.subplot(133),plt.imshow(b_channel,'gray'),plt.title('B')
    plt.tight_layout()
    plt.savefig('static/UPLOAD_FOLDER_WO_VIGENERE/citra_uji2_split.png')
    
def enkripsi(kunci, r_channel ,g_channel,b_channel):

    time_start = time.perf_counter()
    r_channel_encrypted = encryptImage(r_channel, kunci)
    g_channel_encrypted = encryptImage(g_channel, kunci)
    b_channel_encrypted = encryptImage(b_channel, kunci)

    encrypted_img = cv2.merge((b_channel_encrypted,g_channel_encrypted,r_channel_encrypted))
    time_stop = time.perf_counter()
    time_enkripsi = round(time_stop - time_start,3)
    plt.figure(figsize=(8, 4))
    plt.subplot(231),plt.imshow(r_channel_encrypted,'gray'),plt.title('R (encrypted)')
    plt.subplot(232),plt.imshow(g_channel_encrypted,'gray'),plt.title('G (encrypted)')
    plt.subplot(233),plt.imshow(b_channel_encrypted,'gray'),plt.title('B (encrypted)')
    plt.subplot(235),plt.imshow(cv2.cvtColor(encrypted_img, cv2.COLOR_BGR2RGB)), plt.title('Encrypted Image')
    plt.tight_layout()
    plt.savefig('static/output/enkripsi_img.png')

    plt.figure(figsize=(12, 4))
    plt.subplot(131),plt.hist(r_channel_encrypted.flatten(), bins='auto'),plt.title('R')
    plt.subplot(132),plt.hist(g_channel_encrypted.flatten(), bins='auto'),plt.title('G')
    plt.subplot(133),plt.hist(b_channel_encrypted.flatten(), bins='auto'),plt.title('B')
    plt.tight_layout()
    plt.savefig('static/output/image_enkripsi_histogram.png')

    print('Waktu enkripsi image: {} detik'.format(time_stop - time_start))
    return r_channel_encrypted,g_channel_encrypted,b_channel_encrypted,encrypted_img,time_enkripsi

def dekripsi(kunci,r_channel_encrypted,g_channel_encrypted,b_channel_encrypted):
    time_start = time.perf_counter()
    r_channel_decrypted = decryptImage(r_channel_encrypted, kunci)
    g_channel_decrypted = decryptImage(g_channel_encrypted, kunci)
    b_channel_decrypted = decryptImage(b_channel_encrypted, kunci)
    restored_img = cv2.merge((b_channel_decrypted,g_channel_decrypted,r_channel_decrypted))
    time_stop = time.perf_counter()
    time_dekripsi = round(time_stop-time_start,3)
    plt.figure(figsize=(8, 4))
    plt.subplot(231),plt.imshow(r_channel_decrypted,'gray'),plt.title('R (decrypted)')
    plt.subplot(232),plt.imshow(g_channel_decrypted,'gray'),plt.title('G (decrypted)')
    plt.subplot(233),plt.imshow(b_channel_decrypted,'gray'),plt.title('B (decrypted)')
    plt.subplot(235),plt.imshow(cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)), plt.title('Decrypted Image')
    plt.tight_layout()
    plt.savefig('static/output/dekripsi_img.png')

    # histogramnya
    plt.figure(figsize=(12, 4))
    plt.subplot(131),plt.hist(r_channel_decrypted.flatten(), bins='auto'),plt.title('R')
    plt.subplot(132),plt.hist(g_channel_decrypted.flatten(), bins='auto'),plt.title('G')
    plt.subplot(133),plt.hist(b_channel_decrypted.flatten(), bins='auto'),plt.title('B')
    plt.tight_layout()
    plt.savefig('static/output/image_dekripsi_histogram.png')

    cv2.imwrite('static/output/image_dekripsi.jpg',restored_img)


    print('Waktu dekripsi image: {} detik'.format(time_stop - time_start))
    return restored_img,time_dekripsi