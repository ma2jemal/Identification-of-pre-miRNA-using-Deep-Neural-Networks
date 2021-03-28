""" Construct the CNN model for deep learning
"""

import keras
from keras import regularizers
from keras.models import Sequential, Model
from keras.layers import Dense,Activation,Dropout,Flatten, Input, GRU
from keras.layers import Conv1D,MaxPooling1D
from keras.optimizers import Adam

def CNN_model():
    # model = Sequential()
    # #first layer of convolution and max-pooling
    # model.add(Conv1D(16,4,activation = 'relu',padding = 'same',\
    #           input_shape = (180,12)))
    # model.add(MaxPooling1D(pool_size = 2))
    # #second layer of convolution and max-pooling
    # model.add(Conv1D(32,5,activation = 'relu',padding = 'same'))
    # model.add(MaxPooling1D(pool_size = 2))
    # #third layer of convolution and max-pooling
    # model.add(Conv1D(64,6,activation = 'relu',padding = 'same'))
    # model.add(MaxPooling1D(pool_size = 2))
    #
    # model.add(Flatten())
    # model.add(Dropout(0.3))
    # model.add(Dense(32,activation = 'relu',kernel_regularizer = regularizers.l2(0.1)))
    # model.add(Dropout(0.3))
    # model.add(Dense(2,activation = 'softmax'))
    # adam = Adam()
    # model.compile(loss = 'categorical_crossentropy',optimizer = adam,\
    #           metrics = ['accuracy'])
    # print(model.summary())
    input_layer = Input(shape=(180, 12), )
    # x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Dropout(rate=0.2)(input_layer)
    x = Conv1D(filters=4, kernel_size=2, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=4, kernel_size=2, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=4, kernel_size=2, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = GRU(32)(x)
    x = Dropout(rate=0.3)(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(2, activation="softmax")(x)
    model = Model(inputs=input_layer, outputs=x)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
if __name__ == "__main__":
     model = CNN_model()
