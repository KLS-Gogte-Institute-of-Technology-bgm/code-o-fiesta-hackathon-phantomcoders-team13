#libraries
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
# import sklearn.model_selection import train_test_split

def datasetloader():
    print()
    print("Loading dataset...")
    path = '../dataset'

    class1 = os.path.join(path,'class1')
    class2 = os.path.join(path,'class2')
    class3 = os.path.join(path,'class3')

    x1 = len(os.listdir(class1))
    x2 = len(os.listdir(class2))
    x3 = len(os.listdir(class3))
    checksum = x1+x2+x3
    if(checksum == 2000):
        print("Checkpoint Passed!")

datasetloader()
