
from flask import Flask, render_template, request, redirect,url_for,render_template, request
from werkzeug.utils import secure_filename
import json
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LSTM
from keras.utils import to_categorical
from flask import Flask, redirect, url_for, request, render_template,jsonify, make_response, send_from_directory

from gevent.pywsgi import WSGIServer

# Import Keras dependencies
from tensorflow.keras.models import model_from_json


import numpy as np
import h5py

import os

import librosa

import urllib.request

from flask import Flask, flash, request, redirect, render_template


UPLOAD_FOLDER = './uploads'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = set(['wav'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


MODEL_ARCHITECTURE = 'check2.json'
MODEL_WEIGHTS = 'complete_weight.h5'

json_file = open(MODEL_ARCHITECTURE)
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Get weights into the model
model.load_weights(MODEL_WEIGHTS)
print('Model loaded. Check http://127.0.0.1:5000/')

from werkzeug.utils import secure_filename


def model_predict(audio_data,model):
    audio_data= app.config['UPLOAD_FOLDER']+"/"+audio_data
    x, sr = librosa.load(audio_data)
    mfcc = librosa.feature.mfcc(x, sr)
    mfcc = np.asarray(mfcc)
    mfcc = mfcc.reshape(1, mfcc.shape[1], mfcc.shape[0], 1)
    print(type(mfcc), mfcc.shape)
    model = Sequential()
    model.compile(loss="categorical_crossentropy", optimizer="adam",
				  metrics=['accuracy'])
    prediction = np.argmax(model.predict_classes(mfcc))
    print("pppppppppppppppppppppppppppppppppp",prediction)
    return prediction

@app.route('/')
def upload_file():
	return render_template('upload.html')

@app.route('/uploader', methods=['POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filesaved = file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            checking = model_predict(file.filename,model)
            print("checked value is = ", str(checking))
            return jsonify("The predicted value is {}".format(str(checking)))

        else:
            flash('Allowed file types are wav')
            return redirect(request.url)


if __name__ == '__main__':
	app.run()
