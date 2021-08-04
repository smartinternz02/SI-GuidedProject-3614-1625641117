import numpy as np
from flask import Flask, request, render_template
from joblib import load
import joblib
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer
import keras
import tensorflow as tf
import os
from tensorflow.keras.preprocessing import image
from PIL import ImageFile
model = load_model("animal.h5")

index = ['BEAR','CROW','ELEPHANT','RACOON','RAT']

app = Flask(__name__,template_folder="templates",static_folder="styles")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)


        img = image.load_img(filepath,target_size = (64,64))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis =0)
        preds = model.predict_classes(x)
        print("prediction: ",index[preds[0]])

        return render_template('result.html',animal = index[preds[0]])

if __name__ == "__main__":
    app.run(debug=True)
