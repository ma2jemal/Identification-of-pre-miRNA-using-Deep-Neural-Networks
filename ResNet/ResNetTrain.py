""" Train the CNN model using dataset
"""
import sys
sys.path.append("../data") 
from ResNetModel import ResNet_model
import dataSetPartition
import keras
import pandas as pd
import os


def ResNet_train(x_dataset,y_dataset, str):
    model = ResNet_model()
    if os.path.exists("ResNet_model_preTrained.h5"):
        print("load the weights")
        model.load_weights("CNN_model_preTrained.h5")
    history = model.fit(x_dataset,y_dataset,batch_size = 150, epochs = 400,\
          validation_split = 0.2)
    history_dif = pd.DataFrame(history.history)
    path = str + ".csv"
    history_dif.to_csv(path)
    print("model train over")
    return model

if __name__ == "__main__":
    positive = "../data/hsa_new.csv" 
    negative = "../data/pseudo_new.csv"
    x_train_dataset,y_train_dataset,x_test_dataset,y_test_dataset = \
      dataSetPartition.train_test_partition(positive,negative)
    name = "ResNet_model"
    model = ResNet_train(x_train_dataset,y_train_dataset, name)
    # model.save("ResNet_model_preTrained.h5")
    # print("The model is saved as ResNet_model_preTrained.h5 in the current directory")

