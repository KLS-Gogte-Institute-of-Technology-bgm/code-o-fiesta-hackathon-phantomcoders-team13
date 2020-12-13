# code-o-fiesta-hackathon-phantomcoders-team13
code-o-fiesta-hackathon-phantomcoders-team13 created by GitHub Classroom

# Toll Booth
A Toll Booth based on Machine Learning that works un-manned; to generate QR code encoded amount that the traveller can pay using UPI based on the vehicle it detects along with the number of axles it possesses.

## Overview
**1. The Dataset**
The Dataset is scrapped from the internet as it was not readily available for download.
The website is as follows : https://data.mendeley.com/datasets/9s33gv3gbt/draft?a=d664b4e6-b18b-4bc9-b5e9-e45363f2bb79

The Dataset is originally of 1000 samples. It was customised and hence Data Augmented to 2000 samples with other preprocessing techniques.

The Dataset was further cleaned and manually segregated into 3 classes as follows:
* **class1** : Light weight vehicles (1-axle).
* **class2** : Medium weight vehicles (2-axles).
* **class3** : Heavy weight vehicles (more than 2-axles).

**2.Data Preparation**
    The dataset was then loaded and preprocessed using *Keras*. We used the 80-20 train-test split and it was futhered processed and fed to the model.

**3.Transfer Learning, Fine Tuning & Model Compilation**
We used Transfer Learning with a ResNet50 model that had been pre-trained on the ImageNet Dataset. The model has been trained using a *categorical-crossentropy loss* and optimised using *RMSProp*. *Early Fitting* was used to prevent *Overfitting*.

**4.Deployment**





## Built with
* Python
* CSS
* HTML

## Python Dependencies
* Numpy
* Keras
* Tensorflow
* Plotly
* Flask
* imutils
* OpenCV
* GitHub-lfs : GitHub-lfs is required to successfully clone this repository as the trained machine learning model and it's parameters are over 100MB.

## Authors
* **Nikita Kodkany** - *Machine Learning Practioner*
* **Prathamesh Korannae** - *Machine Learning Practioner*
* **Sujay Amberkar** - *API Developer*
* **Varun Shiri** - *Web Developer*
