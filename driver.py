from utilities.sample_predict import sample_predict
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image

model = load_model(filepath, compile=True)

samples_to_predict = sample_predict()
from sample in samples_to_predict:
    prediction = model.predict(sample)
    print(prediction)
