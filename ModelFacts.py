
import pandas as pd
# import sys
# sys.path.append("./data")
# from data import dataSetPartition
# from keras.models import load_model

def ModelFacts():
    ResNet_model_path = "ResNet/ResNetModel.csv"
    FCN_model_path = "FCN/FCNModel.csv"

    ResNet = pd.read_csv(ResNet_model_path)
    FCN = pd.read_csv(FCN_model_path)

    #plot ResNe Loss val loss

    val1 = ResNet.loc[:, ['loss', 'val_loss']].plot().get_figure()
    val2 = ResNet.loc[:, ['acc', 'val_acc']].plot().get_figure()
    val1.savefig("ResNet_valoss")
    val2.savefig("ResNet_accloss")

    val3 = FCN.loc[:, ['loss', 'val_loss']].plot().get_figure()
    val4 = FCN.loc[:, ['acc', 'val_acc']].plot().get_figure()
    val3.savefig("FCN_valoss")
    val4.savefig("FCN_accloss")

if __name__  == "__main__":
    ModelFacts()
