""" Plotting the ROC curve of CNN and RNN  on test dataset
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import plot_confusion_matrix
import scikitplot as skplt


#from sklearn import cross_validation
import sys
sys.path.append("./data")
from data import dataSetPartition
from keras.models import load_model

def PerformancePlot():
    positive = "data/hsa_new.csv"
    negative = "data/pseudo_new.csv"
    CNN_model_path = "CNN/CNN_model.h5"
    RNN_model_path = "RNN/RNN_model.h5"
    ResNet_model_path = "ResNet/ResNet_model.h5"
    FCN_model_path = "FCN/FCN_model.h5"


    x_train_dataset,y_train_dataset,x_test_dataset,y_test_dataset = \
            dataSetPartition.train_test_partition(positive,negative)
    print("load the model")
    try:
        CNN_model = load_model(CNN_model_path)
        RNN_model = load_model(RNN_model_path)
        ResNet_model = load_model(ResNet_model_path)
        FCN_model = load_model(FCN_model_path)
    except Exception:
        print("The model file doesn't exist!")
        exit(1)
    cnn_predict_result = CNN_model.predict(x_test_dataset)
    rnn_predict_result = RNN_model.predict(x_test_dataset)
    resnet_predict_result = ResNet_model.predict(x_test_dataset)
    fcn_predict_result = FCN_model.predict(x_test_dataset)

    # print(predict_result)

    # Compute ROC curve and ROC area for each class
    cnn_fpr,cnn_tpr,cnn_threshold = roc_curve(y_test_dataset[:,0],cnn_predict_result[:,0]) 
    rnn_fpr,rnn_tpr,rnn_threshold = roc_curve(y_test_dataset[:,0],rnn_predict_result[:,0])
    resnet_fpr, resnet_tpr, resnet_threshold = roc_curve(y_test_dataset[:, 0], resnet_predict_result[:, 0])
    fcn_fpr, fcn_tpr, fcn_threshold = roc_curve(y_test_dataset[:, 0], fcn_predict_result[:, 0])



    ## calculate the AUC value
    cnn_roc_auc = auc(cnn_fpr,cnn_tpr) 
    rnn_roc_auc = auc(rnn_fpr,rnn_tpr)
    resnet_roc_auc = auc(resnet_fpr, resnet_tpr)
    fcn_roc_auc = auc(fcn_fpr, fcn_tpr)

    # plotting
    plt.figure(figsize=(10,10))
    plt.plot(cnn_fpr, cnn_tpr, '-',\
         linewidth=2, label='CNN model-AUC:%0.4f)' %cnn_roc_auc)
    plt.plot(resnet_fpr, resnet_tpr, '--', \
             linewidth=2, label='ResNet model-AUC:%0.4f)' % resnet_roc_auc)
    plt.plot(rnn_fpr, rnn_tpr, '--',\
         linewidth=2, label='RNN model-AUC:%0.4f)' %rnn_roc_auc)

    plt.plot(fcn_fpr, fcn_tpr, '--', \
             linewidth=2, label='FCN model-AUC:%0.4f)' % fcn_roc_auc)

   # plt.plot([0, 1], [0, 1], color='navy', linewidth=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
   # plt.title('Receiver operating characteristic')
    plt.legend(loc = "center right")
    plt.savefig("ROC_curve.png",dpi=600)
    plt.show()

    titles = [("ResNet"), ("FCN")]
    for title in titles:
       plot_confusion_matrix(ResNet_model,x_test_dataset, y_test_dataset)
       plt.savefig(title + ".png", dpi = 600)
       plt.show()



if __name__  == "__main__":
    PerformancePlot()
