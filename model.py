#libraries
import tensorflow as tf
import tensorflow.keras as K
from utilities.datasetloader import *
import plotly.graph_objects as go
from keras.models import Model
import pickle


print("\n Tensorflow Version: ",tf.__version__)

train, test, img_width, img_height = datasetloader()

#Transfer learning
input_t = K.Input(shape=(img_width, img_height,3))
resnet50 = K.applications.ResNet50(include_top=False, weights='imagenet',input_tensor=input_t)
output = resnet50.layers[-1].output
output = K.layers.Flatten()(output)
resnet50 = Model(resnet50.input, outputs=output)
for layer in resnet50.layers:
	layer.trainable = False

resnet50.summary()
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
model.add(K.layers.Dense(3, activation='softmax'))

print("\n ---Model Summary---")
model.summary()

loss = tf.keras.losses.CategoricalCrossentropy(from_logits = False)

print("\n Model compiling...")
model.compile(loss=loss, optimizer=K.optimizers.RMSprop(lr=2e-5), metrics=['accuracy'])

print("\n Training...")
history = model.fit(
    train,
    batch_size=16,
    epochs=10,
    validation_data = test,
    validation_steps = 1,
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.01,
        patience=7
    )])

model.save('.')
pickle.dump(model, open('model.pkl','wb'))

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
