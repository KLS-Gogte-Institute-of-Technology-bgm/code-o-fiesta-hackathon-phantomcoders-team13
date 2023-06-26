#libraries
import tensorflow as tf
import tensorflow.keras as K
from utilities.datasetloader import *
import plotly.graph_objects as go
from keras.models import Model
from keras.utils.vis_utils import plot_model



print("\n Tensorflow Version: ",tf.__version__)

train, test, img_width, img_height = datasetloader()
input_t = K.Input(shape=(img_width, img_height,3))
from keras.utils.vis_utils import plot_model

vgg16 = tf.keras.applications.VGG16( include_top=False, weights='imagenet', input_tensor=input_t)

output = vgg16.layers[-1].output
output = tf.keras.layers.Flatten()(output)

vgg16 = Model(vgg16.input, outputs=output)
for layer in vgg16.layers:
    layer.trainable = False

model_vgg16 = tf.keras.models.Sequential()
model_vgg16.add(vgg16)
model_vgg16.add(tf.keras.layers.Flatten())
model_vgg16.add(tf.keras.layers.BatchNormalization())
model_vgg16.add(tf.keras.layers.Dense(256, activation='relu'))
model_vgg16.add(tf.keras.layers.Dropout(0.5))
model_vgg16.add(tf.keras.layers.BatchNormalization())
model_vgg16.add(tf.keras.layers.Dense(128, activation='relu'))
model_vgg16.add(tf.keras.layers.Dropout(0.5))
model_vgg16.add(tf.keras.layers.BatchNormalization())
model_vgg16.add(tf.keras.layers.Dense(64, activation='relu'))
model_vgg16.add(tf.keras.layers.Dropout(0.5))
model_vgg16.add(tf.keras.layers.BatchNormalization())
model_vgg16.add(tf.keras.layers.Dense(3, activation='softmax'))

# #Transfer learning
input_t = K.Input(shape=(img_width, img_height,3))
# resnet50 = K.applications.ResNet50(include_top=False, weights='imagenet',input_tensor=input_t)
# output = resnet50.layers[-1].output
# output = K.layers.Flatten()(output)
# resnet50 = Model(resnet50.input, outputs=output)
# for layer in resnet50.layers:
# 	layer.trainable = False

# resnet50.summary()
# # #To freeze some layers (if needed)
# # for layer in resnet50.layers[:143]:
# #   layer.trainable = False

# # #Print all the layers
# # print("")
# # for i, layer in enumerate(resnet50.layers):
# #   print(i, layer.name,'-',layer.trainable)

# #To connect the pretrained model with new layers (if needed )
# model = K.models.Sequential()
# model.add(resnet50)
# model.add(K.layers.Flatten())
# model.add(K.layers.BatchNormalization())
# model.add(K.layers.Dense(256, activation='relu'))
# model.add(K.layers.Dropout(0.5))
# model.add(K.layers.BatchNormalization())
# model.add(K.layers.Dense(128, activation='relu'))
# model.add(K.layers.Dropout(0.5))
# model.add(K.layers.BatchNormalization())
# model.add(K.layers.Dense(64, activation='relu'))
# model.add(K.layers.Dropout(0.5))
# model.add(K.layers.BatchNormalization())
# model.add(K.layers.Dense(3, activation='softmax'))

print("\n ---Model Summary---")
model_vgg16.summary()
plot_model(model_vgg16, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

loss = tf.keras.losses.CategoricalCrossentropy(from_logits = False)

print("\n Model compiling...")
model_vgg16.compile(loss=loss, optimizer=K.optimizers.RMSprop(lr=2e-5), metrics=['accuracy'])

print("\n Training...")
history = model_vgg16.fit(
    train,
    batch_size=16,
    epochs=20,
    validation_data = test,
    validation_steps = 1,
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.01,
        patience=7
    )])

#model.save('.')
# pickle.dump(model, open('model.pkl','wb'))

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
