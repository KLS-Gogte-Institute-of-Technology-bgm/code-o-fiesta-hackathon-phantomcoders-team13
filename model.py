#libraries
import tensorflow as tf
import tensorflow.keras as K

tf.compat.v1.disable_v2_behavior()
# print("\n Tensorflow Version: ",tf.__version__)

#Transfer learning
input_t = K.Input(shape=(32,32,3))
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

#weight checkpoints
check_point = K.callbacks.ModelCheckpoint(filepath="cifar10.h5",monitor="val_acc", mode="max", save_best_only=True,)

print("\n Model compiling...")
model.compile(loss='categorical_crossentropy', optimizer=K.optimizers.RMSprop(lr=2e-5), metrics=['accuracy'])

print("\n Training...")
history = model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1, validation_data=(x_test, y_test), callbacks=[check_point])

print("\n ---Model Summary---")
model.summary()
