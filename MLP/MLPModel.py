""" Construct the MLP model for deep learning
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

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np


def MLP_model():

    input_layer = keras.layers.Input((180, 12))

    # flatten/reshape because when multivariate all should be on the same axis
    input_layer_flattened = keras.layers.Flatten()(input_layer)

    layer_1 = keras.layers.Dropout(0.1)(input_layer_flattened)
    layer_1 = keras.layers.Dense(64, activation='relu')(layer_1)

    layer_2 = keras.layers.Dropout(0.2)(layer_1)
    layer_2 = keras.layers.Dense(128, activation='relu')(layer_2)

    layer_3 = keras.layers.Dropout(0.2)(layer_2)
    layer_3 = keras.layers.Dense(256, activation='relu')(layer_3)

    output_layer = keras.layers.Dropout(0.3)(layer_3)
    output_layer = keras.layers.Dense(2, activation='softmax')(output_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    model.summary()
    return model
if __name__ == "__main__":
     model = MLP_model()


