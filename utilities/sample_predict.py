import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image

def sample_predict(img ):

    #img_path = "./images/image.jpg"

    img = image.load_img(img, target_size=(180,180))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    samples_to_predict=[]
    samples_to_predict.append(img_preprocessed)

    return samples_to_predict
sample_predict()
