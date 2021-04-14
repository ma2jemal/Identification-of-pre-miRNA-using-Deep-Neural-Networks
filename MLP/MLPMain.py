"""Train the dataset and evaluate the performance on the test dataset using 
   different partition of dataset. The model and the performance results are 
   stored as files.
"""

import sys
sys.path.append("../data")
from MLPTrain import MLP_train
from MLPEvaluation import test_evaluation
import dataSetPartition
import time

def MLPMain():
    # positive and negative dataset
    positive = "../data/hsa_new.csv"
    negative = "../data/pseudo_new.csv"
    # partition the whole dataset into train(4/5) and test(1/5) datasets
    x_train_dataset,y_train_dataset,x_test_dataset,y_test_dataset = \
        dataSetPartition.train_test_partition(positive,negative)
    # train the model

    model = MLP_train(x_train_dataset,y_train_dataset)
    model_path = "MLP_model.h5"
    model.save(model_path)
    print("The model is saved as",model_path,"in the current directory.")

    # evaluate the performance
    sensitivity,specifity,f1_score,mcc,accuracy =\
        test_evaluation(model_path,x_test_dataset,y_test_dataset)
    write_to_file(model_path,sensitivity,specifity,accuracy,f1_score,mcc)

    # Partition the whole dataset into 5 segments with every one segment for test and
    # and the remaining nine segments for train. All the data are stored in four list
    x_train_list,y_train_list,x_validation_list,y_validation_list = \
        dataSetPartition.fold5_cv_partition(positive,negative)
    sen = []
    spec = []
    acc = []
    f1 = []
    mc = []

    m = len(x_train_list)
    for i in range(m):
        model = MLP_train(x_train_list[i],y_train_list[i])
        model_path = "MLP_model_5fold"+str(i)+".h5"
        model.save(model_path)
        print(model_path,"is stored in the current directory.")
        # evaluate the performance
        sensitivity,specifity,f1_score,mcc,accuracy =\
            test_evaluation(model_path,x_validation_list[i],y_validation_list[i])
        sen.append(sensitivity)
        spec.append(specifity)
        acc.append(accuracy)
        f1.append(f1_score)
        mc.append(mcc)
        # write to file
        write_to_file(model_path,sensitivity,specifity,accuracy,f1_score,mcc)
    write_to_file("MLP_model_performanceAVG", avg(sen), avg(spec), avg(acc), avg(f1), avg(mc))

def write_to_file(model_path,sensitivity,specifity,accuracy,f1_score,mcc):
    """ write the performace parameters to file
    """
    fd = open("MLP_model_performance","a+")
    fd.write(str(time.time())+model_path + "performance:")
    fd.write("\n")
    fd.write("sensitivity:{}\n".format(sensitivity))
    fd.write("specifity:{}\n".format(specifity))
    fd.write("accuracy:{}\n".format(accuracy))
    fd.write("f1_score:{}\n".format(f1_score))
    fd.write("mcc:{}\n".format(mcc))
    fd.write("\n\n")
    fd.close()
    
def avg(a):
    return round(sum(a)/len(a),2)

if __name__ == "__main__":
    start = time.time()
    MLPMain()
    end = time.time()
    print("finished! and it took " + str(end-start))
