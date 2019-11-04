from cProfile import label

from PIL import Image
import cv2
import numpy as np
import stack_wrapper
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

# TODO Add comments, test the model with different layers and filters



# load train Data
Train_X = np.zeros((30,512,512),dtype=np.int16)
Train_Y = np.zeros((30,512,512),dtype=np.int16)
sw_data  = stack_wrapper.Stack_wrapper('Data/train-volume.tif')
sw_lable = stack_wrapper.Stack_wrapper('Data/train-labels.tif')


for i in range(30) :
    Train_X[i] = sw_data.next()
    Train_Y[i] = sw_lable.next()

# normalizing data
Train_X = Train_X/255.0
Train_X = np.expand_dims(Train_X,axis=-1)
Train_Y = Train_Y/255.0
Train_Y = np.expand_dims(Train_Y,axis=-1)




def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
    return c, p

def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    us = keras.layers.UpSampling2D((2, 2))(x)
    concat = keras.layers.Concatenate()([us, skip])
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c


def UNet():
    filters = [16, 32, 64, 128, 256]
    inputs = keras.layers.Input((512, 512, 1))

    p0 = inputs
    c1, p1 = down_block(p0, filters[0])  # 128 -> 64
    c2, p2 = down_block(p1, filters[1])  # 64 -> 32
    c3, p3 = down_block(p2, filters[2])  # 32 -> 16
    c4, p4 = down_block(p3, filters[3])  # 16->8

    bn = bottleneck(p4, filters[4])

    u1 = up_block(bn, c4, filters[3])  # 8 -> 16
    u2 = up_block(u1, c3, filters[2])  # 16 -> 32
    u3 = up_block(u2, c2, filters[1])  # 32 -> 64
    u4 = up_block(u3, c1, filters[0])  # 64 -> 128

    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u4)
    model = keras.models.Model(inputs, outputs)
    return model

def plot_history(histories) :
    plt.figure(figsize=(16,10))

    for name, history in histories :

        val = plt.plot(history.epoch,history.history['val_loss'],
                       '--',label=name.title()+' Val')
        plt.plot(history.epoch,history.history['loss'],
                 color=val[0].get_color(),label=name.title()+' Train')


    plt.xlabel('Epochs')
    plt.ylabel('Binary Crossentropy')
    plt.legend()
    plt.xlim([0,max(history.epoch)])
    plt.show()



model = UNet()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
model.summary()
history = model.fit(Train_X,Train_Y,validation_split=0.2,epochs=50,batch_size=1)
print(history.history.keys())
plot_history([('modified U-Net',history)])


model.save_weights("UNetWeights.h5")

# load test Data
Test_X = np.zeros((30,512,512),dtype=np.int16)
sw_data  = stack_wrapper.Stack_wrapper('Data/test-volume.tif')
for i in range(30) :
    Test_X[i] = sw_data.next()

Test_X = Test_X/255.0
Test_X = np.expand_dims(Test_X,axis=-1)

result = model.predict(Test_X)

result = result > 0.5

np.save('result.npy',result)

fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)

ax = fig.add_subplot(1, 2, 1)
ax.imshow(np.reshape(Test_X[0]*255, (512, 512)), cmap="gray")

ax = fig.add_subplot(1, 2, 2)
ax.imshow(np.reshape(result[0]*255, (512, 512)), cmap="gray")

plt.show()

