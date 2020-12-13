import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image

model = load_model(filepath, compile=True)

def sample_predict():

    img1_path = "./dataset/class1/8"
    img2_path = "./dataset/class2/240"
    img3_path = "./dataset/class3/1226"
    img4_path = "./dataset/class1/8"
    # img5_path = "./dataset/class2/1176"
    # img6_path = "./dataset/class3/1635"
    # img7_path = "./dataset/class1/300"

    img1 = image.load_img(img1_path, target_size=(180,180))
    img2 = image.load_img(img2_path, target_size=(180,180))
    img3 = image.load_img(img3_path, target_size=(180,180))
    img4 = image.load_img(img4_path, target_size=(180,180))
    # img5 = image.load_img(img5_path, target_size=(180,180))
    # img6 = image.load_img(img6_path, target_size=(180,180))
    # img7 = image.load_img(img7_path, target_size=(180,180))

    img1_array = image.img_to_array(img1)
    img1_batch = np.expand_dims(img1_array, axis=0)

    img2_array = image.img_to_array(img2)
    img2_batch = np.expand_dims(img2_array, axis=0)

    img3_array = image.img_to_array(img3)
    img3_batch = np.expand_dims(img3_array, axis=0)

    img4_array = image.img_to_array(img4)
    img4_batch = np.expand_dims(img4_array, axis=0)

    img1_preprocessed = preprocess_input(img1_batch)
    img2_preprocessed = preprocess_input(img2_batch)
    img3_preprocessed = preprocess_input(img3_batch)
    img4_preprocessed = preprocess_input(img4_batch)

    samples_to_predict=[]
    samples_to_predict.append(img1_preprocessed)
    samples_to_predict.append(img2_preprocessed)
    samples_to_predict.append(img3_preprocessed)
    samples_to_predict.append(img4_preprocessed)

    return samples_to_predict
sample_predict()
