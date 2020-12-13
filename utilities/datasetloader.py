#libraries
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
# import sklearn.model_selection import train_test_split

def datasetloader():
    print()
    batch_size = 1
    img_height = 180
    img_width = 180

    print("Loading dataset...")
    PATH = '../'
    path = '../dataset'

    # print(os.listdir(PATH))
    dataset = os.path.join(PATH,'dataset')

    class1 = os.path.join(path,'class1')
    class2 = os.path.join(path,'class2')
    class3 = os.path.join(path,'class3')

    x1 = len(os.listdir(class1))
    x2 = len(os.listdir(class2))
    x3 = len(os.listdir(class3))
    checksum = x1+x2+x3
    if(checksum == 2000):
        print("Checkpoint 1 Passed!")
        print()

    train = tf.keras.preprocessing.image_dataset_from_directory(
        dataset,
        validation_split = 0.1,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    print()

    test = tf.keras.preprocessing.image_dataset_from_directory(
        dataset,
        validation_split=0.1,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    print()
    return train, test, img_width, img_height
datasetloader()
