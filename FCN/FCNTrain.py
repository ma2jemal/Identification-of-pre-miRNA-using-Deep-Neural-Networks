""" Train the CNN model using dataset
"""
import sys
import pandas as pd
sys.path.append("../data") 
from FCNModel import FCN_model
import dataSetPartition
import keras
import os


def FCN_train(x_dataset,y_dataset, str):
    model = FCN_model()
    if os.path.exists("FCN_model_preTrained.h5"):
        print("load the weights")
        model.load_weights("FCN_model_preTrained.h5")
    history = model.fit(x_dataset,y_dataset,batch_size = 200, epochs = 600,\
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
    name = "FCN_model"
    model = FCN_train(x_train_dataset,y_train_dataset, name)




