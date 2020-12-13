#libraries
import tensorflow as tf
import tensorflow.keras as K
from utilities import *
import plotly.graph_objects as go

print("\n Tensorflow Version: ",tf.__version__)

train, test, img_width, img_height = datasetloader()

#Transfer learning
input_t = K.Input(shape=(img_width, img_height,3))
resnet50 = K.applications.ResNet50(include_top=50, weights='imagenet',input_tensor=input_t)

# #To freeze some layers (if needed)
# for layer in resnet50.layers[:143]:
#   layer.trainable = False

# #Print all the layers
# print("")
# for i, layer in enumerate(resnet50.layers):
#   print(i, layer.name,'-',layer.trainable)

#To connect the pretrained model with new layers (if needed )
model = K.models.Sequential()
model.add(resnet50)
model.add(K.layers.Flatten())
model.add(K.layers.BatchNormalization())
model.add(K.layers.Dense(256, activation='relu'))
model.add(K.layers.Dropout(0.5))
model.add(K.layers.BatchNormalization())
model.add(K.layers.Dense(128, activation='relu'))
model.add(K.layers.Dropout(0.5))
model.add(K.layers.BatchNormalization())
model.add(K.layers.Dense(64, activation='relu'))
model.add(K.layers.Dropout(0.5))
model.add(K.layers.BatchNormalization())
model.add(K.layers.Dense(10, activation='softmax'))

print("\n ---Model Summary---")
model.summary()

print("\n Model compiling...")
model.compile(loss='categorical_crossentropy', optimizer=K.optimizers.RMSprop(lr=2e-5), metrics=['accuracy'])

print("\n Training...")
history = model.fit(
    train,
    batch_size=1,
    epochs=10,
    validation_data = test,
    validation_steps = 1,
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.01,
        patience=7
    )]


fig = go.Figure()
fig.add_trace(go.Scatter(x=history.epoch,
                         y=history.history['accuracy'],
                         mode='lines+markers',
                         name='Training accuracy'))
fig.add_trace(go.Scatter(x=history.epoch,
                         y=history.history['val_accuracy'],
                         mode='lines+markers',
                         name='Validation accuracy'))
fig.update_layout(title='Accuracy',
                  xaxis=dict(title='Epoch'),
                  yaxis=dict(title='Percentage'))
fig.show()
