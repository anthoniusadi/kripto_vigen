
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import os

# pip install pycryptodome
def check_size(path_file):
    value_size = os.stat(path_file)
    # MB size
    value_size = value_size.st_size / (1024*1024)
    return round(value_size,4)


def enkripsi(key,path_image_original):
    # x : path image
    time_start = time.perf_counter()
    global  image_encrypted_B,image_encrypted_G,image_encrypted_R, merging_encr
    num_list = [ord(x) - 96 for x in key]
    mean = np.mean(num_list)
    std = np.std(num_list)
    image_input = cv2.imread(path_image_original)
    # cv2.imwrite('static/output/original.jpg',image_input)
    b_channel ,g_channel, r_channel = cv2.split(image_input)
    # cv2.imwrite('static/output/b_channel.jpg',b_channel)
    # cv2.imwrite('static/output/g_channel.jpg',g_channel)
    # cv2.imwrite('static/output/r_channel.jpg',r_channel)
    x1, y,c = image_input.shape
    key = np.random.normal(mean, std, (x1, y))
#     image_encrypted_B = ((b_channel / key) * np.mean(b_channel) )* 255
#     image_encrypted_G = ((g_channel / key) * np.mean(g_channel) ) * 255
#     image_encrypted_R = ((r_channel / key) * np.mean(r_channel) )* 255

    image_encrypted_B = (b_channel / key) * 255
    image_encrypted_G = (g_channel / key) * 255
    image_encrypted_R = (r_channel / key) * 255
    merging_encr=cv2.merge((image_encrypted_B,image_encrypted_G,image_encrypted_R))
    time_stop = time.perf_counter()
    time_enkripsi = round(time_stop - time_start,3)


    cv2.imwrite('static/output/image_encrypted_R.jpg', image_encrypted_R )
    cv2.imwrite('static/output/image_encrypted_G.jpg', image_encrypted_G )
    cv2.imwrite('static/output/image_encrypted_B.jpg', image_encrypted_B )
    cv2.imwrite('static/output/enkripsi_img.jpg', merging_encr )


    r = cv2.imread('static/output/image_encrypted_R.jpg')
    g = cv2.imread('static/output/image_encrypted_G.jpg')
    b = cv2.imread('static/output/image_encrypted_B.jpg')
    enkrip_image = cv2.imread('static/output/enkripsi_img.jpg')
    print('buat plot')
    plt.figure(figsize=(8, 4))
    plt.subplot(231),plt.imshow(r),plt.title('R (encrypted)')
    plt.subplot(232),plt.imshow(g),plt.title('G (encrypted)')
    plt.subplot(233),plt.imshow(b),plt.title('B (encrypted)')
    # plt.subplot(233),plt.imshow(enkrip_image),plt.title('(encrypted_image)')
    plt.subplot(235),plt.imshow(cv2.cvtColor(enkrip_image, cv2.COLOR_BGR2RGB)),plt.title('(encrypted_image)')

    # plt.subplot(235),plt.imshow(cv2.cvtColor(merging_encr, cv2.COLOR_BGR2RGB)), plt.title('Encrypted Image')
    plt.tight_layout()
    plt.savefig('static/output/enkripsi_img.png')
    print('buat histo2')

    # tmp = cv2.imread('static/output/enkripsi_img.jpg')
    b = cv2.calcHist([enkrip_image],[0],None,[255],[0,255])
    g = cv2.calcHist([enkrip_image],[1],None,[255],[0,255])
    r = cv2.calcHist([enkrip_image],[2],None,[255],[0,255])    


    plt.figure(figsize=(13, 7))
    plt.subplot(231),plt.plot(r),plt.title('R (encrypted)')
    plt.xlim(0, 255)
    plt.subplot(232),plt.plot(g),plt.title('G (encrypted)')
    plt.xlim(0, 255)
    plt.subplot(233),plt.plot(b),plt.title('B (encrypted)')
    plt.xlim(0, 255)
    # plt.subplot(131),plt.hist(image_encrypted_R, bins='auto'),plt.title('R')
    # plt.subplot(132),plt.hist(image_encrypted_G, bins='auto'),plt.title('G')
    # plt.subplot(133),plt.hist(image_encrypted_B, bins='auto'),plt.title('B')
    plt.tight_layout()
    plt.savefig('static/output/image_enkripsi_histogram.png')

    print('Encrypted successfully! Waktu enkripsi image: {} detik'.format(time_stop - time_start))
    return image_encrypted_R,image_encrypted_G,image_encrypted_B,merging_encr,time_enkripsi,key

def dekripsi(key, r_channel, g_channel, b_channel):
#     global image_encrypted
    time_start = time.perf_counter()

    # b_channel ,g_channel, r_channel = cv2.split(merging)
    image_decrypted_B = (b_channel * key )/ 255
    image_decrypted_G = (g_channel * key )/ 255
    image_decrypted_R = (r_channel * key ) / 255

    merging_decr=cv2.merge((image_decrypted_B,image_decrypted_G,image_decrypted_R))
    time_stop = time.perf_counter()
    time_dekripsi = round(time_stop - time_start,3)
    cv2.imwrite('static/output/image_dekripsi.jpg', merging_decr)
    dekrip_image = cv2.imread('static/output/image_dekripsi.jpg')

    plt.figure(figsize=(8, 4))
    plt.subplot(231),plt.imshow(image_decrypted_R,'gray'),plt.title('R (decrypted)')
    plt.subplot(232),plt.imshow(image_decrypted_G,'gray'),plt.title('G (decrypted)')
    plt.subplot(233),plt.imshow(image_decrypted_B,'gray'),plt.title('B (decrypted)')
    plt.subplot(235),plt.imshow(cv2.cvtColor(dekrip_image, cv2.COLOR_BGR2RGB),'gray'),plt.title('(decrypted_image)')

    # plt.subplot(235),plt.imshow(cv2.cvtColor(merging_decr, cv2.COLOR_BGR2RGB)), plt.title('Decrypted Image')
    # plt.subplot(235),plt.imshow(merging_decr), plt.title('Decrypted Image')

    plt.tight_layout()
    plt.savefig('static/output/dekripsi_img.png')
    b = cv2.calcHist([dekrip_image],[0],None,[255],[0,255])
    g = cv2.calcHist([dekrip_image],[1],None,[255],[0,255])
    r = cv2.calcHist([dekrip_image],[2],None,[255],[0,255])    

    # histogramnya
    plt.figure(figsize=(12, 4))
    # plt.subplot(131),plt.hist(image_decrypted_R.flatten(), bins='auto'),plt.title('R')
    # plt.subplot(132),plt.hist(image_decrypted_G.flatten(), bins='auto'),plt.title('G')
    # plt.subplot(133),plt.hist(image_decrypted_B.flatten(), bins='auto'),plt.title('B')
    plt.subplot(131),plt.plot(r),plt.title('R (encrypted)')
    plt.xlim(0, 255)
    plt.subplot(132),plt.plot(g),plt.title('G (encrypted)')
    plt.xlim(0, 255)
    plt.subplot(133),plt.plot(b),plt.title('B (encrypted)')
    plt.xlim(0, 255)
    plt.tight_layout()
    plt.savefig('static/output/image_dekripsi_histogram.png')

    print('Decryted successfully! Waktu enkripsi image: {} detik'.format(time_stop - time_start))

    return time_dekripsi
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
    plt.subplot(131),plt.imshow(cv2.cvtColor(r_channel, cv2.COLOR_BGR2RGB)),plt.title('R')
    plt.subplot(132),plt.imshow(cv2.cvtColor(g_channel, cv2.COLOR_BGR2RGB)),plt.title('G')
    plt.subplot(133),plt.imshow(cv2.cvtColor(b_channel, cv2.COLOR_BGR2RGB)),plt.title('B')
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

# def enkripsi(kunci, r_channel ,g_channel,b_channel):

#     time_start = time.perf_counter()
#     r_channel_encrypted = encryptImage(r_channel, kunci)
#     g_channel_encrypted = encryptImage(g_channel, kunci)
#     b_channel_encrypted = encryptImage(b_channel, kunci)

#     encrypted_img = cv2.merge((b_channel_encrypted,g_channel_encrypted,r_channel_encrypted))
#     time_stop = time.perf_counter()
#     time_enkripsi = round(time_stop - time_start,3)
#     plt.figure(figsize=(8, 4))
#     plt.subplot(231),plt.imshow(r_channel_encrypted,'gray'),plt.title('R (encrypted)')
#     plt.subplot(232),plt.imshow(g_channel_encrypted,'gray'),plt.title('G (encrypted)')
#     plt.subplot(233),plt.imshow(b_channel_encrypted,'gray'),plt.title('B (encrypted)')
#     plt.subplot(235),plt.imshow(cv2.cvtColor(encrypted_img, cv2.COLOR_BGR2RGB)), plt.title('Encrypted Image')
#     plt.tight_layout()
#     plt.savefig('static/output/enkripsi_img.png')

#     plt.figure(figsize=(12, 4))
#     plt.subplot(131),plt.hist(r_channel_encrypted.flatten(), bins='auto'),plt.title('R')
#     plt.subplot(132),plt.hist(g_channel_encrypted.flatten(), bins='auto'),plt.title('G')
#     plt.subplot(133),plt.hist(b_channel_encrypted.flatten(), bins='auto'),plt.title('B')
#     plt.tight_layout()
#     plt.savefig('static/output/image_enkripsi_histogram.png')

#     print('Waktu enkripsi image: {} detik'.format(time_stop - time_start))
#     return r_channel_encrypted,g_channel_encrypted,b_channel_encrypted,encrypted_img,time_enkripsi

# def dekripsi(kunci,r_channel_encrypted,g_channel_encrypted,b_channel_encrypted):
#     time_start = time.perf_counter()
#     r_channel_decrypted = decryptImage(r_channel_encrypted, kunci)
#     g_channel_decrypted = decryptImage(g_channel_encrypted, kunci)
#     b_channel_decrypted = decryptImage(b_channel_encrypted, kunci)
#     restored_img = cv2.merge((b_channel_decrypted,g_channel_decrypted,r_channel_decrypted))
#     time_stop = time.perf_counter()
#     time_dekripsi = round(time_stop-time_start,3)
#     plt.figure(figsize=(8, 4))
#     plt.subplot(231),plt.imshow(r_channel_decrypted,'gray'),plt.title('R (decrypted)')
#     plt.subplot(232),plt.imshow(g_channel_decrypted,'gray'),plt.title('G (decrypted)')
#     plt.subplot(233),plt.imshow(b_channel_decrypted,'gray'),plt.title('B (decrypted)')
#     plt.subplot(235),plt.imshow(cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)), plt.title('Decrypted Image')
#     plt.tight_layout()
#     plt.savefig('static/output/dekripsi_img.png')
#     # histogramnya
#     plt.figure(figsize=(12, 4))
#     plt.subplot(131),plt.hist(r_channel_decrypted.flatten(), bins='auto'),plt.title('R')
#     plt.subplot(132),plt.hist(g_channel_decrypted.flatten(), bins='auto'),plt.title('G')
#     plt.subplot(133),plt.hist(b_channel_decrypted.flatten(), bins='auto'),plt.title('B')
#     plt.tight_layout()
#     plt.savefig('static/output/image_dekripsi_histogram.png')

#     cv2.imwrite('static/output/image_dekripsi.jpg',restored_img)
#     print('Waktu dekripsi image: {} detik'.format(time_stop - time_start))
#     return restored_img,time_dekripsi