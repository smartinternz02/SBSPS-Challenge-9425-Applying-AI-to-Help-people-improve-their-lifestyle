import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model=load_model('BrainTumor10EpochsCategorical.h5')

image=cv2.imread('C:\\Users\\HP\\Documents\\BrainTumor Classification DL\\pred\\pred7.jpg')

img=Image.fromarray(image)

img=img.resize((64,64))

img=np.array(img)


input_img=np.expand_dims(img, axis= 0)

result1=model.predict(input_img)
result = np.argmax(result1,axis=1)
print(result)




