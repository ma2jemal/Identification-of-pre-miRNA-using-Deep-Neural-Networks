import matplotlib
import matplotlib.pyplot as plt
import numpy as np

senstivity =0.7376623376623377
Sns_lst =[0.7526315789473684, 0.9076923076923077, 0.8238636363636364, 0.8808290155440415, 0.89, 0.8300970873786407, 0.8722222222222222, 0.8806818181818182,0.7303921568627451, 0.8074534161490683]

Specify = 0.9564032697547684
spc_lst=[0.9408602150537635, 0.8950276243093923, 0.955, 0.7814207650273224, 0.9147727272727273, 0.9529411764705882, 0.8979591836734694, 0.9, 0.9418604651162791,0.9539170506912442]

Accuracy = 0.8444148936170213
Acc_lst=[0.8457446808510638, 0.90159574468085, 0.8324468085106383, 0.901595744680851, 0.8856382978723404, 0.8856382978723404, 0.8856382978723404, 0.8909574468085106, 0.8271276595744681, 0.8915343915343915]

F1score =0.8291970802919709
f1_lst= [0.8313953488372093, 0.9053708439897699, 0.8883116883116884, 0.8795518207282913, 0.8831908831908831,0.8209366391184573, 0.8637873754152824,0.8795518207282913, 0.8883116883116884, 0.905852417302799]

Mcc = 0.7084905741914618
Mcc_lst =[0.7050789538003446, 0.802890560111184, 0.7903071517649632, 0.6665954286572315, 0.8034158264790621, 0.7803356118059301, 0.7708043311709706, 0.7809591967012056, 0.6779277052927474, 0.7796260380735538]
Csensivity = 0.9064935064935065

CSns_lst=[0.8894736842105263,0.8256410256410256, 0.8693181818181818, 0.8290155440414507, 0.895, 0.8689320388349514, 0.9055555555555556, 0.8579545454545454, 0.8774509803921569, 0.9254658385093167]


Cspcfty = 0.8610354223433242
Cspc_lst=[0.8817204301075269, 0.9613259668508287, 0.91, 0.9344262295081968, 0.875, 0.9411764705882353, 0.9081632653061225, 0.945, 0.8604651162790697, 0.8894009216589862]

CAccuracy = 0.8843085106382979
CAcc_lst=[ 0.8856382978723404, 0.8909574468085106, 0.8909574468085106, 0.8803191489361702, 0.8856382978723404, 0.901595744680851, 0.9069148936170213, 0.9042553191489362,  0.8696808510638298, 0.9047619047619048]

CF1score =0.889171974522293
Cf1_lst= [0.8871391076115485, 0.8870523415977961, 0.8818443804034583, 0.8767123287671232, 0.8927680798004988, 0.9063291139240506, 0.9030470914127424, 0.893491124260355,0.8796068796068796, 0.8922155688622754]


CMcc = 0.7688769170662344
CMcc_lst =[0.7712486754355518, 0.7909095181035936, 0.7809282877110559, 0.7659506903470428, 0.7702735832383361, 0.8063982183344651, 0.8135458915597865, 0.809091984518306, 0.7375903406797488, 0.8087781789980407]


def ave(lst):
    return round((sum(lst)/len(lst))*100)
def per(num):
    return round(num * 100)
rnn = []
rnn.append(ave(Sns_lst))
rnn.append(ave(spc_lst))
rnn.append(ave(Acc_lst))
rnn.append(ave(f1_lst))
rnn.append(ave(Mcc_lst))

labels = ['CNN', 'CNN(10-fold)', 'RNN', 'RNN(10-fold)']
sensti = [per(Csensivity), ave(CSns_lst), per(senstivity), ave(Sns_lst)]
speci= [per(Cspcfty), ave(Cspc_lst), per(Specify), ave(spc_lst)]
accuracy= [per(CAccuracy), ave(CAcc_lst), per(Accuracy), ave(Acc_lst)]
f1 = [per(CF1score), ave(Cf1_lst), per(F1score), ave(f1_lst)]
mc = [per(CMcc), ave(CMcc_lst), per(Mcc), ave(Mcc_lst)]

width = 0.15  # the width of the bars
x1 = np.arange(len(labels))  # the label locations
x2 = [i+width for i in x1]
x3 = [i+width for i in x2]
x4 = [i+width for i in x3]
x5 = [i+width for i in x4]

fig, ax = plt.subplots()


rects1 = ax.bar(x1, sensti, width)
rects2 = ax.bar(x2, speci, width)
rects3 = ax.bar(x3, accuracy, width)
rects4 = ax.bar(x4, f1, width)
rects5 = ax.bar(x5, mc, width)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Performance(%)')
ax.set_title('Performance of the proposed models')
ax.set_xticks(x1+ 0.21)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
autolabel(rects5)

fig.tight_layout()
fig.savefig("Peformance_plot.jpeg")
plt.show()









