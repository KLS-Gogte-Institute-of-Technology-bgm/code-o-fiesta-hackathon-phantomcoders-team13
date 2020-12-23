from utilities.sample_predict import sample_predict
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from utility import qrgen
import cv2

model = tf.keras.models.load_model('.')

samples_to_predict = sample_predict('./images/image.jpg')
for sample in samples_to_predict:
    prediction = model.predict(sample)
    predict = np.argmax(prediction)
    print("\nclass" + str(np.argmax(prediction)+1))

    if predict == 0:
    	print("Amount is 100")
    	qrgen(100)

    elif predict ==1:
    	print("Amount is 200")	
    	qrgen(200)
    else:
    	print("Amount is 300 ")
    	qrgen(300)

    img = cv2.imread('static/sample.png')

    cv2.imshow('',img)

    cv2.waitKey(0)


cv2.destroyAllWindows()





