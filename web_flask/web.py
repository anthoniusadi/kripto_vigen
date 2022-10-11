

# from optparse import Values
# from pickle import TRUE
from glob import glob
from flask import Flask,redirect,render_template,request,url_for
import os
import modules
from modules.cleaning import rename
import cv2

from Crypto.Cipher import AES
import io
import PIL.Image
import binascii
import math


modules.create_folder()
# cleaning.create_folder()
app = Flask(__name__)

FOLDER1 = os.path.join('static' , 'UPLOAD_FOLDER_VIGENERE')
FOLDER2 = os.path.join('static' , 'UPLOAD_FOLDER_WO_VIGENERE')
TEMP_FOLDER = os.path.join('static' , 'TEMP_FOLDER')
OUT_FOLDER = os.path.join('static' , 'output')

app.config['UPLOAD_FOLDER1'] = FOLDER1
app.config['UPLOAD_FOLDER2'] = FOLDER2
app.config['TEMP_FOLDER'] = TEMP_FOLDER
app.config['OUT_FOLDER'] = OUT_FOLDER
state = False
state_process=False
vigen_state = True
@app.route('/',methods=['GET','POST'])
def dashboard():

    return render_template('dashboard.html')

@app.route('/check',methods=["GET","POST"])
def check():
    selected_menu = request.form.get('vigenere')
    if selected_menu == 'yes':
        vigen_state = True
        return redirect(url_for('process'))
    vigen_state = False
    return redirect(url_for('non_vigen'))



@app.route('/process', methods = ["POST"])
def process():
    selected_menu = request.form.get('vigenere')
    if selected_menu == 'yes':
        vigen_state = True
    else:
        vigen_state = False

    if vigen_state:
        global filename1
        global filename2
        global filename3
        global filename4
        global filename5
        global filename6
        global state_process
        global time_enkrip
        global time_dekrip
        values = []
        state_process=True
        if request.method == 'POST':
            file1 = request.files['imgfile']
            name_img = request.files.getlist('imgfile')
            # file1 = request.files('imgfile')
            path = os.path.join(app.config['TEMP_FOLDER'], 'original.jpg')
            # tmp = modules.rename(file1)
            file1.save(path)
            key = request.form['key']
            read = cv2.imread(path)
            outfile = 'original' + '.jpg'
            cv2.imwrite('static/output/'+outfile,read,[cv2.IMWRITE_JPEG_QUALITY, 100])

        # menampilkan original image
        filename1 = os.path.join(app.config['TEMP_FOLDER'], 'original.jpg')
        # split dalam RGB chanell
        b,g,r = modules.histogram(filename1)

        # menampilkan citra encrypted
        # do encrypt
        r_encrypted,g_encrypted,b_encrypted,img_encrypted,time_enkrip,key = modules.enkripsi(key=key,path_image_original=filename1)

        time_dekrip = modules.dekripsi(key=key,r_channel= r_encrypted,g_channel= g_encrypted,b_channel= b_encrypted)
        filename1 = os.path.join(app.config['OUT_FOLDER'], 'original.jpg')
        filename2 = os.path.join(app.config['OUT_FOLDER'], 'image_dekripsi_histogram.png')
        filename3 = os.path.join(app.config['OUT_FOLDER'], 'enkripsi_img.png')
        filename4 = os.path.join(app.config['OUT_FOLDER'], 'dekripsi_img.png')
        filename5 = os.path.join(app.config['OUT_FOLDER'], 'image_enkripsi_histogram.png')
        filename6 = os.path.join(app.config['OUT_FOLDER'], 'image_dekripsi.jpg')

        return render_template('dashboard.html',filename1=filename1,filename2=filename2 ,filename3 = filename3,filename4=filename4,filename5=filename5,filename6=filename6,state_process=state_process,values=values,state=state)

    if request.method == 'POST':
        file1 = request.files['imgfile']
        name_img = request.files.getlist('imgfile')
        # file1 = request.files('imgfile')
        path = os.path.join(app.config['TEMP_FOLDER'], 'original.jpg')
        # tmp = modules.rename(file1)
        file1.save(path)
        key = request.form['key']
        read = cv2.imread(path)
        outfile = 'original' + '.jpg'
        cv2.imwrite('static/output/'+outfile,read,[cv2.IMWRITE_JPEG_QUALITY, 100])

    # menampilkan original image
    filename1 = os.path.join(app.config['TEMP_FOLDER'], 'original.jpg')
    # split dalam RGB chanell
    b,g,r = modules.histogram(filename1)

    # menampilkan citra encrypted
    # do encrypt
    r_encrypted,g_encrypted,b_encrypted,img_encrypted,time_enkrip = modules.enkripsi(kunci=key,r_channel=r,g_channel=g,b_channel=b)
    img_decrypt,time_dekrip = modules.dekripsi(key,r_encrypted,g_encrypted,b_encrypted)
    filename1 = os.path.join(app.config['OUT_FOLDER'], 'original.jpg')
    filename2 = os.path.join(app.config['OUT_FOLDER'], 'image_dekripsi_histogram.png')
    filename3 = os.path.join(app.config['OUT_FOLDER'], 'enkripsi_img.png')
    filename4 = os.path.join(app.config['OUT_FOLDER'], 'dekripsi_img.png')
    filename5 = os.path.join(app.config['OUT_FOLDER'], 'image_enkripsi_histogram.png')
    filename6 = os.path.join(app.config['OUT_FOLDER'], 'image_dekripsi.jpg')

    return render_template('dashboard.html',filename1=filename1,filename2=filename2 ,filename3 = filename3,filename4=filename4,filename5=filename5,filename6=filename6,state_process=state_process,values=values,state=state,time_enkrip=time_enkrip,time_dekrip=time_dekrip)

############################################################## non vigenere route ###################################################################
@app.route('/non_vigen', methods = ["GET","POST"])
def non_vigen():
    return render_template('dashboard_non_vigenere.html')

@app.route('/process_non', methods = ["GET","POST"])
def process_non():
    selected_menu = request.form.get('vigenere')
    if selected_menu == 'yes':
        vigen_state = True
    else:
        vigen_state = False

    if (vigen_state == False):

        global state_process
        global citrauji1
        global citrauji2
        global histo1
        global histo2
        global time_non

        values = []
        state_process=True
        if request.method == 'POST':
            # citra 1
            file1 = request.files['imgfile1']
            name_img = request.files.getlist('imgfile1')
            path1 = os.path.join(app.config['TEMP_FOLDER'], 'original1.jpg')

            file1.save(path1)
            read1 = cv2.imread(path1)
            outfile1 = 'original1' + '.jpg'
            cv2.imwrite('static/UPLOAD_FOLDER_WO_VIGENERE/'+outfile1,read1)
            # citra 2
            file2 = request.files['imgfile2']
            name_img = request.files.getlist('imgfile2')
            path2 = os.path.join(app.config['TEMP_FOLDER'], 'original2.jpg')

            file2.save(path2)
            read2 = cv2.imread(path2)
            outfile2 = 'original2' + '.jpg'
            cv2.imwrite('static/UPLOAD_FOLDER_WO_VIGENERE/'+outfile2,read2)

        citrauji1 = os.path.join(app.config['TEMP_FOLDER'], 'original1.jpg')
        citrauji2 = os.path.join(app.config['TEMP_FOLDER'], 'original2.jpg')
        # split dalam RGB chanell
        modules.histogram_non(citrauji1,citrauji2)
        # b,g,r = modules.histogram_non(citrauji2,citrauji2)



        # menampilkan original image

        citrauji1 = os.path.join(app.config['UPLOAD_FOLDER2'], 'original1.jpg')
        citrauji2 = os.path.join(app.config['UPLOAD_FOLDER2'], 'original2.jpg')
        histo1 = os.path.join(app.config['UPLOAD_FOLDER2'], 'citra_uji1_histogram.png')
        histo2 = os.path.join(app.config['UPLOAD_FOLDER2'], 'citra_uji2_histogram.png')
        time_non = "Tidak ada komputasi enkripsi dan dekripsi"


        return render_template('dashboard_non_vigenere.html',citrauji1=citrauji1,citrauji2=citrauji2,histo1=histo1,histo2=histo2,state_process=state_process,values=values,state=state,time_non = time_non)
    # else:

    #     if request.method == 'POST':
    #         file1 = request.files['imgfile']
    #         name_img = request.files.getlist('imgfile')
    #         # file1 = request.files('imgfile')
    #         path = os.path.join(app.config['TEMP_FOLDER'], 'original.jpg')
    #         # tmp = modules.rename(file1)
    #         file1.save(path)
    #         key = request.form['key']
    #         read = cv2.imread(path)
    #         outfile = 'original' + '.jpg'
    #         cv2.imwrite('static/UPLOAD_FOLDER_WO_VIGENERE/'+outfile,read,[int(cv2.IMWRITE_JPEG_QUALITY),200])

    #     # menampilkan original image
    #     filename1 = os.path.join(app.config['TEMP_FOLDER'], 'original.jpg')
    #     # split dalam RGB chanell
    #     b,g,r = modules.histogram(filename1)

    #     # menampilkan citra encrypted
    #     # do encrypt
    #     r_encrypted,g_encrypted,b_encrypted,img_encrypted = modules.enkripsi(kunci=key,r_channel=r,g_channel=g,b_channel=b)
    #     img_decrypt = modules.dekripsi(key,r_encrypted,g_encrypted,b_encrypted)
    #     filename1 = os.path.join(app.config['OUT_FOLDER'], 'original.jpg')
    #     filename2 = os.path.join(app.config['OUT_FOLDER'], 'image_dekripsi_histogram.png')
    #     filename3 = os.path.join(app.config['OUT_FOLDER'], 'enkripsi_img.png')
    #     filename4 = os.path.join(app.config['OUT_FOLDER'], 'dekripsi_img.png')
    #     filename5 = os.path.join(app.config['OUT_FOLDER'], 'image_enkripsi_histogram.png')
    #     filename6 = os.path.join(app.config['OUT_FOLDER'], 'image_dekripsi.jpg')

    #     return render_template('dashboard_non_vigenere.html',filename1=filename1,filename2=filename2 ,filename3 = filename3,filename4=filename4,filename5=filename5,filename6=filename6,state_process=state_process,values=values,state=state)


@app.route('/evaluasi')
def evaluasi():
    state=True
    if state_process :
        source_img = cv2.imread(filename1)
        restored_img = cv2.imread(filename6)
        print(filename1,filename6)
        values = [modules.calc_psnr(source_img,restored_img),modules.calc_mse(source_img,restored_img),modules.npcr(source_img,restored_img),modules.uaci(source_img,restored_img),(modules.entropy(restored_img)),time_enkrip,time_dekrip,modules.check_size(filename1),modules.check_size(filename6)]

        tmp_val = f'PSNR : {modules.calc_psnr(source_img,restored_img)}db\nMSE : {modules.calc_mse(source_img,restored_img)}\nNPCR : {modules.npcr(source_img,restored_img)} %\nUACI : {modules.uaci(source_img,restored_img)}\nENTROPY : {modules.entropy(restored_img)}\nWaktu Komputasi Enkripsi : {time_enkrip} detik\nWaktu Komputasi Dekripsi : {time_dekrip} detik \n Image_size citra uji : {modules.check_size(filename1)}Mb \n Image_size citra dekripsi : {modules.check_size(filename6)}Mb '

        with open('static/output/evaluation.txt','w') as out:
            out.write(tmp_val)
        ket = 'Output stored in static/output folder'
        return render_template('dashboard.html',filename1=filename1,filename2=filename2 ,filename3 = filename3,filename4=filename4,filename5=filename5,filename6=filename6,values=values,keterangan=ket,state_process=state_process,state=state,time_enkrip=time_enkrip,time_dekrip=time_dekrip)
    # return redirect('/')
    return render_template('dashboard.html')
# non vigenere
@app.route('/evaluasi_non')
def evaluasi_non():
    state=True
    if state_process :
        source_img = cv2.imread(citrauji1)
        restored_img = cv2.imread(citrauji2)
        values = [modules.calc_psnr(source_img,restored_img),modules.calc_mse(source_img,restored_img),modules.npcr(source_img,restored_img),modules.uaci(source_img,restored_img),(modules.entropy(restored_img)),time_non]

        tmp_val = f'PSNR : {modules.calc_psnr(source_img,restored_img)}db\nMSE : {modules.calc_mse(source_img,restored_img)}\nNPCR : {modules.npcr(source_img,restored_img)} %\nUACI " {modules.uaci(source_img,restored_img)}\nENTROPY : {modules.entropy(restored_img)}\n {time_non} '

        with open('static/UPLOAD_FOLDER_WO_VIGENERE/evaluation.txt','w') as out:
            out.write(tmp_val)
        ket = 'Output stored in static/UPLOAD_FOLDER_WO_VIGENERE folder'
        return render_template('dashboard_non_vigenere.html',citrauji1=citrauji1,citrauji2=citrauji2,histo1=histo1,histo2=histo2,state_process=state_process,values=values,state=state,time_non=time_non)
    # return redirect('/')
    return render_template('dashboard_non_vigenere.html')
if __name__ == '__main__':
    app.run(debug=True)