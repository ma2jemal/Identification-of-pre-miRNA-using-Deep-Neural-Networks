""" Train the MLP model using dataset
"""
import sys
sys.path.append("../data") 
from MLPModel import MLP_model
import dataSetPartition
import keras
import os


def MLP_train(x_dataset,y_dataset):
    model = MLP_model()
    if os.path.exists("MLP_model_preTrained.h5"):
        print("load the weights")
        model.load_weights("MLP_model_preTrained.h5")
    model.fit(x_dataset,y_dataset,batch_size = 150, epochs = 200,\
          validation_split = 0.2)
    print("model train over")
    return model

if __name__ == "__main__":
    positive = "../data/hsa_new.csv" 
    negative = "../data/pseudo_new.csv"
    x_train_dataset,y_train_dataset,x_test_dataset,y_test_dataset = \
      dataSetPartition.train_test_partition(positive,negative)
    model = MLP_train(x_train_dataset,y_train_dataset)
    model.save("MLP_model_preTrained.h5")
    print("The model is saved as MLP_model_preTrained.h5 in the current directory")

