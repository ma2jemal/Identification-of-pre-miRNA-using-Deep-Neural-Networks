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


def CNN_model():
    n_feature_maps = 64

    input_layer = keras.layers.Input((180, 12))

    # conv block -1
    conv_1 = keras.layers.Conv1D(filters=5, kernel_size=5, activation='relu', padding='same')(input_layer)
    conv_1 = keras.layers.MaxPool1D(pool_size=2)(conv_1)

    conv_2 = keras.layers.Conv1D(filters=20, kernel_size=5, activation='relu', padding='same')(conv_1)
    conv_2 = keras.layers.MaxPool1D(pool_size=4)(conv_2)

    flatten_layer = keras.layers.Flatten()(conv_2)
    fully_connected_layer = keras.layers.Dense(200, activation='relu')(flatten_layer)

    output_layer = keras.layers.Dense(2, activation='softmax')(fully_connected_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    model.compile(optimizer=keras.optimizers.Adam(lr=0.01, decay=0.005),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model
if __name__ == "__main__":
     model = CNN_model()


