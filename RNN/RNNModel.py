""" Construct the RNN model for deep learning
"""

import keras
from keras import regularizers
from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense, BatchNormalization
from keras.layers import LSTM,Masking
from keras.optimizers import Adam



def RNN_model():
    # SEG_LENTH = 180
    # model = Sequential()
    # model.add(Masking(mask_value= [0,0,0,0,0,0,0,0,0,0,0,0],\
    #          input_shape=(SEG_LENTH, 12)))
    # model.add(LSTM(128,dropout=0.2, recurrent_dropout=0.2,\
    #               # kernel_regularizer = regularizers.l2(0.1),\
    #                input_shape = (SEG_LENTH, 12),return_sequences = True))
    # model.add(LSTM(64,dropout=0.2, recurrent_dropout=0.2,\
    #               # kernel_regularizer = regularizers.l2(0.1),
    #                return_sequences = True))
    # model.add(LSTM(2))
    # model.add(Activation('softmax'))
    # adam = Adam()
    # model.compile(loss = 'categorical_crossentropy',optimizer = adam,\
    #               metrics = ['accuracy'])
    model = Sequential()
    model.add(LSTM(32,input_shape=(180,12),recurrent_dropout=0.3,return_sequences=True ))
    model.add(LSTM(64, return_sequences=True, recurrent_dropout=0.3, ))
    model.add(LSTM(64))
    model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1))
    model.add(Dropout(0.3))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1))
    model.add(Dropout(0.3))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    print(model.summary())
    return model 

if __name__ == "__main__":
     model = RNN_model()
