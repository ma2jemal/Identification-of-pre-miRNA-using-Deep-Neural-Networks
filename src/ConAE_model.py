import keras
from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D
from keras.models import Model

def ConAE():

    numberOfRows = 1881 + 1881
    input_sig = Input(shape=(180,12))
    L1 = Conv1D(16, 4, activation='relu' , padding = 'same')(input_sig)
    L2 = MaxPooling1D(pool_size=2, padding='same')(L1)
    L3 = Conv1D(32,4 ,activation='relu', padding = 'same')(L2)
    L4 = MaxPooling1D(pool_size=2, padding='same')(L3)#encoded

    L5 = Conv1D(32, 4, activation='relu', padding='same')(L4)
    L6 = UpSampling1D(size=2)(L5)
    L7 = Conv1D(32,4, activation='relu', padding='same')(L6)
    L8 = UpSampling1D(size=2)(L7)
    Lf = Conv1D(2, 1, activation='sigmoid', padding='same')(L8)

    autoencoder = Model(input_sig, Lf)
    autoencoder.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy'])
    print(autoencoder.summary())

    return autoencoder

if __name__ == "__main__":
    ConAE_model = ConAE()


