from __future__ import division, print_function
# coding=utf-8
import sys
import os
import numpy as np

# Keras
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image

from lime import lime_image
from skimage.segmentation import mark_boundaries

import matplotlib.pyplot as plt

# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'D:/CREAR_APLICACIONES/04_CUARTA_SEMANA/deep_learning_Chest_X_Ray/server/models/Inception_V3'
TARGET_SIZE = (299, 299)
CLASS_0 = 'NORMAL'
CLASS_1 = 'PNEUMONIA'
UMBRAL = 0.5
# Load your trained model

model = None
explainer = lime_image.LimeImageExplainer()

def get_model():
    global model
    if model is None:
        model = load_model(MODEL_PATH)
    return model

def get_explainer():
    global explainer
    if explainer is None:
        explainer = lime_image.LimeImageExplainer()
    return explainer

def convert_image_to_array(img_path):
    img = image.load_img(img_path, target_size=TARGET_SIZE)
    x = image.img_to_array(img)
    return x

def predict_image_array(img_array):
    global model
    model = get_model()
    if len(img_array.shape) >= 4:
        img_array = preprocess_input(img_array)
        return model.predict(img_array)
    img_array = preprocess_input(img_array)
    img_array = img_array.reshape(1, img_array.shape[0], img_array.shape[1], img_array.shape[2])
    return model.predict(img_array)


def model_predict(img_path):
    global model
    model = get_model()
    img = image.load_img(img_path, target_size=TARGET_SIZE)
    # Preprocessing the image
    x = image.img_to_array(img)
    return predict_image_array(x)


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    global model
    model = get_model()
    if request.method == 'POST':
        f = request.files['image']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        img_array = convert_image_to_array(file_path)
        preds = predict_image_array(img_array)
        # explanation = explainer.explain_instance(img_array, predict_image_array)
        pred_class = CLASS_0 if preds[0][0] < UMBRAL else CLASS_1
        # temp, mask = explanation.get_image_and_mask(explanation.top_labels[0],
        #                                             positive_only=True,
        #                                             num_features=5,
        #                                             hide_rest=False)
        # img = plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
        # img.set_cmap('hot')
        # plt.axis('off')
        # # plt.Axes.remove()
        # # plt.show()
        # plt.savefig(file_path + '_decision.png')
        # return render_template('index.html', clase=pred_class, prob=preds[0][0], image_file=file_path + '_decision.png')
        return '''Clase predicha {}. Probabilidad de tener neumonía {}%'''.format(str(pred_class),
                                                                                  str(round(100 * preds[0][0], 2)))
    return None


get_model()

get_explainer()


if __name__ == '__main__':
    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
