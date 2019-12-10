import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

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


def UNet(filters,lr):

    inputs = keras.layers.Input((512, 512, 1))

    p0 = inputs
    c1, p1 = down_block(p0, filters[0])
    c2, p2 = down_block(p1, filters[1])
    c3, p3 = down_block(p2, filters[2])
    c4, p4 = down_block(p3, filters[3])

    bn = bottleneck(p4, filters[4])

    u1 = up_block(bn, c4, filters[3])
    u2 = up_block(u1, c3, filters[2])
    u3 = up_block(u2, c2, filters[1])
    u4 = up_block(u3, c1, filters[0])

    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u4)
    model = keras.models.Model(inputs, outputs)
    opt =tf.keras.optimizers.SGD( learning_rate=lr, momentum=0.99)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
    model.summary()
    return model



