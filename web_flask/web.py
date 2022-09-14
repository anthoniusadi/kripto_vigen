

from optparse import Values
from pickle import TRUE
from flask import Flask,redirect,render_template,request
import os
import modules
from modules.cleaning import rename
import cv2


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

state_process=False
@app.route('/',methods=['GET','POST'])
def dashboard():
    # renaming file

    return render_template('dashboard.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/process', methods = ["POST"])
def process():
    global filename1
    global filename2
    global filename3
    global filename4
    global state_process
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
        cv2.imwrite('static/output/'+outfile,read,[int(cv2.IMWRITE_JPEG_QUALITY),200])

        print('ini',file1)
    # menampilkan original image
    filename1 = os.path.join(app.config['TEMP_FOLDER'], 'original.jpg')
    # split dalam RGB chanell

    b,g,r = modules.histogram(filename1)

    # menampilkan citra encrypted
    # do encrypt
    r_encrypted,g_encrypted,b_encrypted,img_encrypted = modules.enkripsi(kunci=key,r_channel=r,g_channel=g,b_channel=b)
    img_decrypt = modules.dekripsi(key,r_encrypted,g_encrypted,b_encrypted)
    filename1 = os.path.join(app.config['OUT_FOLDER'], 'original.jpg')
    
    filename2 = os.path.join(app.config['OUT_FOLDER'], 'enkripsi_histogram_img.png')
    filename3 = os.path.join(app.config['OUT_FOLDER'], 'enkripsi_img.png')
    filename4 = os.path.join(app.config['OUT_FOLDER'], 'dekripsi_img.png')

    return render_template('dashboard.html',filename1=filename1,filename2=filename2 ,filename3 = filename3,filename4=filename4,state_process=state_process,values=values)

@app.route('/show')
def show():
    pass


@app.route('/evaluasi')
def evaluasi():
    if state_process:
        source_img = cv2.imread(filename3)
        restored_img = cv2.imread(filename4)
        values = [modules.calc_psnr(source_img,restored_img),modules.calc_mse(source_img,restored_img),modules.npcr(source_img,restored_img),modules.uaci(source_img,restored_img),(modules.entropy(restored_img))]

        tmp_val = f'PSNR : {modules.calc_psnr(source_img,restored_img)}\nMSE : {modules.calc_mse(source_img,restored_img)}\nNPCR : {modules.npcr(source_img,restored_img)} %\nUACI " {modules.uaci(source_img,restored_img)}\nENTROPY : {modules.entropy(restored_img)} '
        
        with open('static/output/evaluation.txt','w') as out:
            out.write(tmp_val)
        ket = 'Output images and results stored in static/output folder'
        return render_template('dashboard.html',filename1=filename1,filename2=filename2 ,filename3 = filename3,filename4=filename4,values=values,keterangan=ket,state_process=state_process)
    else:
        # return redirect('/')
        return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(debug=True)