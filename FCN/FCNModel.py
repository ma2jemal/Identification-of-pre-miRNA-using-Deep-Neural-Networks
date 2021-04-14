""" Construct the CNN model for deep learning
"""
#
# import keras
# from keras import regularizers
# from keras.models import Sequential, Model
# from keras.layers import Dense,Activation,Dropout,Flatten, Input, GRU, MaxPool1D
# from keras.layers import Conv1D,MaxPooling1D, BatchNormalization
# from keras.optimizers import Adam
# from tensorflow.python.keras.utils.vis_utils import plot_model
# import pydot


import tensorflow.keras as keras
import tensorflow as tf
import numpy as np


def FCN_model():
    n_feature_maps = 64

    input_layer = keras.layers.Input((180, 12))

    conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation(activation='relu')(conv1)

    conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)

    conv3 = keras.layers.Conv1D(128, kernel_size=3, padding='same')(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)

    gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(2, activation='softmax')(gap_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    model.summary()
    return model
if __name__ == "__main__":
     model = FCN_model()


