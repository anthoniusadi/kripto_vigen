import os
import sys
from PIL import Image
import cv2
# do cleaning image data 
'''
image sumber : change -> img_source.format
image encrypt : change -> img_encrypt.jpg
image decrypt : change -> img_decrypt.jpg
image histogram_encrypt : change -> img_histo_encrypt.jpg

'''

def create_folder():
    folder_name = 'static/output'
    exists = os.path.exists(folder_name)
    if not exists:
        os.makedirs(folder_name)
        os.makedirs('static/UPLOAD_FOLDER_VIGENERE')
        os.makedirs('static/UPLOAD_FOLDER_WO_VIGENERE')
        os.makedirs('static/TEMP_FOLDER')
        print(f'folder {folder_name} is created')
def rename(infile):
    new_path = './static/TEMP_FOLDER'
    if infile[-4:] == "tiff" or infile[-3:] == "tif" :
        read = cv2.imread(infile)
        outfile = infile.split('.')[0] + '.jpg'
        cv2.imwrite(new_path+outfile,read,[int(cv2.IMWRITE_JPEG_QUALITY), 200])
    # print "is tif or bmp"

        print('ducess',new_path+outfile,read)
