from __future__ import division, print_function
import sys
import os
import glob
import numpy as np
from keras.preprocessing import image 
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras import backend
from tensorflow.keras import backend
import tensorflow as tf
global graph
graph=tf.get_default_graph()
from skimage.transform import resize

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
MODEL_PATH = 'D:\project1.ex\models\BrainTumorDetection.h5'

model = load_model(MODEL_PATH)
      
print('Model loaded. Check http://127.0.0.1:5000/')

@app.route('/',methods=[ 'GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        img = image.load_img(file_path, target_size=(64, 64))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        
        with graph.as_default():
            preds = model.predict_classes(x)
            2
            print(preds)
        if preds==[1]:
            x='base2.html'
        else:         
            x='base1.html'
    return render_template(x) 

    
if __name__ == '__main__':
    app.run(debug=True,use_reloader=False, threaded = False)


