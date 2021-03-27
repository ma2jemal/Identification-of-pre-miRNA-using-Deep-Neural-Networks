""" Train Model.py using dataset
"""
import sys
sys.path.append("../data") 
from model import ConAE
import dataSetPartition
import keras
import os


def Train(x_dataset,y_dataset):
    model = ConAE()
    if os.path.exists("CNN_model_preTrained.h5"):
        print("load the weights")
        model.load_weights("CNN_model_preTrained.h5")
    model.fit(x_dataset,y_dataset,batch_size = 200, epochs = 600,\
          validation_split = 0.2)
    print("Model.py train over")
    return model

if __name__ == "__main__":
    positive = "../data/hsa_new.csv" 
    negative = "../data/pseudo_new.csv"
    x_train_dataset,y_train_dataset,x_test_dataset,y_test_dataset = \
      dataSetPartition.train_test_partition(positive,negative)
    model = Train(x_train_dataset,y_train_dataset)
#   Model.py.save("CNN_model_preTrained.h5")
#   print("The Model.py is saved as CNN_model_preTrained.h5 in the current directory")

