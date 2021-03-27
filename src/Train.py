""" Train ConAE_model.py using dataset
"""
import sys
sys.path.append("../data") 
from ConAE_model import ConAE
import dataSetPartition
import keras
import os


def Train(x_dataset,y_dataset):
    model = ConAE()
    # if os.path.exists("CNN_model_preTrained.h5"):
    #     print("load the weights")
    #     model.load_weights("CNN_model_preTrained.h5")
    print(x_dataset.shape)
    model.fit(x_dataset,y_dataset,batch_size = 200, epochs = 600,\
          validation_split = 0.2)
    print("ConAE_model.py train over")
    return model

if __name__ == "__main__":
    positive = "../data/hsa_new.csv" 
    negative = "../data/pseudo_new.csv"
    x_train_dataset,y_train_dataset,x_test_dataset,y_test_dataset = \
      dataSetPartition.train_test_partition(positive,negative)
    model = Train(x_train_dataset,y_train_dataset)
#   ConAE_model.py.save("CNN_model_preTrained.h5")
#   print("The ConAE_model.py is saved as CNN_model_preTrained.h5 in the current directory")

