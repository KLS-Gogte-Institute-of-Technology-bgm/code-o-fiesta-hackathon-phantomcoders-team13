from utilities.sample_predict import sample_predict
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image


model = tf.keras.models.load_model('.')

samples_to_predict = sample_predict()
for sample in samples_to_predict:
    prediction = model.predict(sample)
    print("class" + str(np.argmax(prediction)+1))
