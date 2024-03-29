import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import cv2 


def sample_predict(img_path):

    

    img = image.load_img(img_path, target_size=(180,180))
   # img = cv2.resize(img, (180,180))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    samples_to_predict=[]
    samples_to_predict.append(img_preprocessed)



    return samples_to_predict

