from math import sqrt
import os
import optparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

parser = optparse.OptionParser()
parser.add_option('-i', '--image_source',
    action="store", dest="image_source",
    help="path image yang digunakan", default="Model : None \nplease insert image_path" )
parser.add_option('-k', '--kunci',
    action="store", dest="kunci",
    help="kunci yang digunakan", default="Model : None \nplease kunci with string type" )
options, args = parser.parse_args()

def histogram(source_img):
    # img = cv2.imread(source_img_path)
    b_channel ,g_channel, r_channel = cv2.split(source_img)
    plt.figure(figsize=(12, 4))
    plt.subplot(131),plt.hist(r_channel.flatten(), bins='auto'),plt.title('R')
    plt.subplot(132),plt.hist(g_channel.flatten(), bins='auto'),plt.title('G')
    plt.subplot(133),plt.hist(b_channel.flatten(), bins='auto'),plt.title('B')
    plt.tight_layout()
    plt.savefig('output/image_source_histogram.png')

    plt.figure(figsize=(12, 8))
    plt.subplot(131),plt.imshow(r_channel,'gray'),plt.title('R')
    plt.subplot(132),plt.imshow(g_channel,'gray'),plt.title('G')
    plt.subplot(133),plt.imshow(b_channel,'gray'),plt.title('B')
    plt.tight_layout()
    plt.savefig('output/image_source_split.png')

    return b_channel,g_channel,r_channel

def create_folder():
    folder_name = 'output'
    exists = os.path.exists(folder_name)
    if not exists:
        os.makedirs(folder_name)
        print(f'folder {folder_name} is created')

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

def calc_mse(img1,img2):
    error_pixel = (img1 -img2) ** 2
    summed_error = np.sum(error_pixel)
    total_pixel = img1.shape[0] * img1.shape[1] 
    mse_val = summed_error / total_pixel
    return mse_val
def entropy(img1):
        img = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)    

        lensig=img.size
        symset=list(set(img))
        numsym=len(symset)
        propab=[np.size(img[img==i])/(1.0*lensig) for i in symset]
        ent=np.sum([p*np.log2(1.0/p) for p in propab])
        return ent
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

def enkripsi(kunci, r_channel ,g_channel,b_channel):

    time_start = time.perf_counter()
    r_channel_encrypted = encryptImage(r_channel, kunci)
    g_channel_encrypted = encryptImage(g_channel, kunci)
    b_channel_encrypted = encryptImage(b_channel, kunci)

    encrypted_img = cv2.merge((b_channel_encrypted,g_channel_encrypted,r_channel_encrypted))
    time_stop = time.perf_counter()

    plt.figure(figsize=(8, 4))
    plt.subplot(231),plt.imshow(r_channel_encrypted,'gray'),plt.title('R (encrypted)')
    plt.subplot(232),plt.imshow(g_channel_encrypted,'gray'),plt.title('G (encrypted)')
    plt.subplot(233),plt.imshow(b_channel_encrypted,'gray'),plt.title('B (encrypted)')
    plt.subplot(235),plt.imshow(cv2.cvtColor(encrypted_img, cv2.COLOR_BGR2RGB)), plt.title('Encrypted Image')
    plt.tight_layout()
    plt.savefig('output/enkripsi_img.png')

    plt.figure(figsize=(12, 4))
    plt.subplot(131),plt.hist(r_channel_encrypted.flatten(), bins='auto'),plt.title('R')
    plt.subplot(132),plt.hist(g_channel_encrypted.flatten(), bins='auto'),plt.title('G')
    plt.subplot(133),plt.hist(b_channel_encrypted.flatten(), bins='auto'),plt.title('B')
    plt.tight_layout()
    plt.savefig('output/image_enkripsi_histogram.png')
    print('Waktu enkripsi image: {} detik'.format(time_stop - time_start))
    return r_channel_encrypted,g_channel_encrypted,b_channel_encrypted,encrypted_img

def dekripsi(kunci,r_channel_encrypted,g_channel_encrypted,b_channel_encrypted):
    time_start = time.perf_counter()
    r_channel_decrypted = decryptImage(r_channel_encrypted, kunci)
    g_channel_decrypted = decryptImage(g_channel_encrypted, kunci)
    b_channel_decrypted = decryptImage(b_channel_encrypted, kunci)
    restored_img = cv2.merge((b_channel_decrypted,g_channel_decrypted,r_channel_decrypted))
    time_stop = time.perf_counter()

    #save each channel dekripsi 
    plt.figure(figsize=(8, 4))
    plt.subplot(231),plt.imshow(r_channel_decrypted,'gray'),plt.title('R (decrypted)')
    plt.subplot(232),plt.imshow(g_channel_decrypted,'gray'),plt.title('G (decrypted)')
    plt.subplot(233),plt.imshow(b_channel_decrypted,'gray'),plt.title('B (decrypted)')
    plt.subplot(235),plt.imshow(cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)), plt.title('Decrypted Image')
    plt.tight_layout()
    plt.savefig('output/dekripsi_img.png')

    # save restores_img
    cv2.imwrite('output/image_dekripsi.jpg',restored_img)

    print('Waktu dekripsi image: {} detik'.format(time_stop - time_start))
    return restored_img
# untuk evaluasi entropy

def entropy():
    pass

def eval(source_img,restored_img):
    # plt.figure(figsize=(8, 4))
    tmp_val = f'PSNR : {calc_psnr(source_img,restored_img)}\nMSE : {calc_mse(source_img,restored_img)}\nNPCR : {npcr(source_img,restored_img)} %\nUACI : {uaci(source_img,restored_img)} \nENTROPY : {entropy(restored_img)}'
    # plt.subplot(231),plt.imshow(cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)), plt.title('original Image')
    # plt.subplot(232),plt.imshow(cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)), plt.title('restored Image')
    # plt.savefig('')
    with open('output/evaluation.txt','w') as out:
        out.write(tmp_val)
    print(tmp_val)

def main(image_source,kunci):
    create_folder()
    # read and save histogram image source
    source_img = cv2.imread(image_source)
    # bgr
    b_channel,g_channel,r_channel = histogram(source_img)
    # enkripsi and dekripsi 
    r_channel_encrypted,g_channel_encrypted,b_channel_encrypted,encrypted_img = enkripsi(kunci,r_channel,g_channel,b_channel)
    restore_img = dekripsi(kunci,r_channel_encrypted,g_channel_encrypted,b_channel_encrypted)
    # evaluasi PSNR,MSE,NPCR,UACI store in txt file
    # ganti image yang mau diuji dengan memasukan variable
    '''
    keterangan : 
    encrypted_img : hasil image yang sudah di enkripsi
    restore_img: hasil image yang sudah di deksripsi
    source_img : image original
    '''
    eval(source_img,encrypted_img)
    # eval(source_img,restore_img)


if __name__ == "__main__":
    # perhatikan petik dua dan petik satunya
    # how to run -> python main.py -i "1.citra asli.tif" -k 'rahasia'
    # 
    start = time.time()
    print("run")
    main(options.image_source,options.kunci)
    print(f"All process done in {time.time()-start} seconds")


# img1 = cv2.imread("1.citra asli.tif")
# img2 = cv2.imread("17.superimposed.tif")
# eval(img1,img2)