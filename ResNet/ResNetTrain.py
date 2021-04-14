""" Train the CNN model using dataset
"""
import sys
sys.path.append("../data") 
from ResNetModel import ResNet_model
import dataSetPartition
import keras
import os


def ResNet_train(x_dataset,y_dataset):
    model = ResNet_model()
    if os.path.exists("ResNet_model_preTrained.h5"):
        print("load the weights")
        model.load_weights("CNN_model_preTrained.h5")
    model.fit(x_dataset,y_dataset,batch_size = 150, epochs = 400,\
          validation_split = 0.2)
    print("model train over")
    return model

if __name__ == "__main__":
    positive = "../data/hsa_new.csv" 
    negative = "../data/pseudo_new.csv"
    x_train_dataset,y_train_dataset,x_test_dataset,y_test_dataset = \
      dataSetPartition.train_test_partition(positive,negative)
    model = ResNet_train(x_train_dataset,y_train_dataset)
    model.save("ResNet_model_preTrained.h5")
    print("The model is saved as ResNet_model_preTrained.h5 in the current directory")
