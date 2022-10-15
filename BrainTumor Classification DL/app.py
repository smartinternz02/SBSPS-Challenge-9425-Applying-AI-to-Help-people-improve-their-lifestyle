import os
import tensorflow as tf
import numpy as np

from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename


app = Flask(__name__)


model =load_model('BrainTumor10Epochscategorical.h5')
print('Model loaded. Check http://127.0.0.1:5000/')


def get_className(classNo):
	if classNo==0:
		return "You don't seem to have brain tumor!!"
	elif classNo==1:
		return "It seems like you have brain tumor"


def getResult(img):
    image=cv2.imread(img)
    img=Image.fromarray(image)
    img=img.resize((64,64))
    img=np.array(img)
    input_img=np.expand_dims(img, axis= 0)
    result1=model.predict(input_img)
    result = np.argmax(result1,axis=1)
    return result


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value=getResult(file_path)
        
        result=get_className(value)
         
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)