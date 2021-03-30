""" Construct the CNN model for deep learning
"""

import keras
from keras import regularizers
from keras.models import Sequential, Model
from keras.layers import Dense,Activation,Dropout,Flatten, Input, GRU, MaxPool1D
from keras.layers import Conv1D,MaxPooling1D, BatchNormalization
from keras.optimizers import Adam
from tensorflow.python.keras.utils.vis_utils import plot_model
import pydot



def CNN_model():
    model = Sequential()

    model.add(Conv1D(16,4, activation='relu', input_shape=(180, 12)))
    model.add(MaxPool1D(pool_size=2))
    model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1))

    model.add(Conv1D(32,5, activation='relu'))
    model.add(MaxPool1D(pool_size=2, ))

    model.add(Flatten())
    model.add(Dropout(0.5))

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4)))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()
    return model
if __name__ == "__main__":
     model = CNN_model()
