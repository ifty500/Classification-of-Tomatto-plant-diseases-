import sys
import os
import glob
import re
import numpy as np

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'Tomato_model_VGG16_2.h5'

# Load your trained model
model = load_model(MODEL_PATH)




def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)

    
    x = np.expand_dims(x, axis=0)


   

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x)

    preds = model.predict(x)
    print(preds)
    result=np.argmax(preds, axis=1)
    
    if result == 0:
        prediction = 'Tomato___Bacterial_spot'
    elif result == 1:
        prediction = 'Tomato___Early_blight'
    elif result == 2:
        prediction = 'Tomato___Late_blight'
    elif result == 3:
        prediction = 'Tomato___Leaf_Mold'
    elif result == 4:
        prediction = 'Tomato___Septoria_leaf_spot'
    elif result == 5:
        prediction = 'Tomato___Spider_mites Two-spotted_spider_mite'
    elif result == 6:
        prediction = 'Tomato___Target_Spot'
    elif result == 7:
        prediction = 'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
    elif result == 8:
        prediction = 'Tomato___Tomato_mosaic_virus'
    elif result == 9:
        prediction = 'Tomato___healthy'
    
    
    return prediction


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
