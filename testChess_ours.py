from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from NN import *
from nn_train import nn_train
from nn_forward import nn_forward
from nn_test import nn_test
from nn_predict import nn_predict
from nn_backward import nn_backward
from nn_applygradient import nn_applygradient
from function import *
import numpy as np
import pickle
import os

def save_variable(v,filename):
    f=open(filename,'wb')
    pickle.dump(v,f)
    f.close()
    return filename

def load_variable(filename):
    f=open(filename,'rb')
    r=pickle.load(f)
    f.close()
    return r

filename = 'krkopt.data'
fr = open(filename)
arrayOLines = fr.readlines()
del arrayOLines[0]
numberOfLines = len(arrayOLines)
numberOfDimension = 6
data = np.zeros((numberOfLines, numberOfDimension), dtype=float)
label = np.zeros((numberOfLines, 2), dtype=float)
for index in range(len(arrayOLines)):
    line = arrayOLines[index]
    listFromLine = line.split(',')
    data[index, 0] = ord(listFromLine[0])-96
    data[index, 1] = ord(listFromLine[1]) - 48
    data[index, 2] = ord(listFromLine[2])-96
    data[index, 3] = ord(listFromLine[3]) - 48
    data[index, 4] = ord(listFromLine[4])-96
    data[index, 5] = ord(listFromLine[5]) - 48
    if listFromLine[6] == 'draw\n':
        label[index,:] = np.array([1,0])
    else:
        label[index,:] = np.array([0,1])

ratioTraining = 0.4
ratioValidation = 0.1
ratioTesting = 0.5
xTraining, xTesting, yTraining, yTesting = train_test_split(data, label, test_size=1 - ratioTraining, random_state=1)  # 随机分配数据集
xTesting, xValidation, yTesting, yValidation = train_test_split(xTesting, yTesting, test_size=ratioValidation / ratioTesting, random_state=1)

scaler = StandardScaler(copy=False)
scaler.fit(xTraining)
scaler.transform(xTraining)
scaler.transform(xTesting)
scaler.transform(xValidation)

nn = NN(
    layer=[6, 20, 20, 2],
    active_function='sigmoid',
    batch_size=100,
    learning_rate=0.01,
    optimization_method='RMSPropNesterov',
    rho=0.9,
    alpha=0.9,
    stability_constant=1e-8,
    batch_normalization=1,
    objective_function='Cross Entropy'
)
epoch = 0
maxAccuracy = 0
totalAccuracy = []
totalCost = []
maxEpoches = 100
for epoch in range(maxEpoches):
#    epoch +=1
    nn.cost.clear()
    nn = nn_train(nn, xTraining, yTraining)
    totalCost.append(sum(nn.cost.values()) / len(nn.cost.values()))
    _, _, accuracy,_ = nn_test(nn, xValidation, yValidation)
    totalAccuracy.append(accuracy)
    if accuracy > maxAccuracy:
        maxAccuracy = accuracy
        save_variable(nn, 'storedNN_Chess_Ours.npz')
    cost = totalCost[-1]
    print('Epoch:',epoch)
    print('Accuracy:',accuracy)
    print('Cost:',totalCost[-1])

if os.path.exists('storedNN_Chess_Ours.npz'):
    storedNN_Chess_Ours = load_variable('storedNN_Chess_Ours.npz')
    wrongs, yPred, accuracy, yOutput = nn_test(storedNN_Chess_Ours, xTesting, yTesting)
    decisionValues = yOutput[:,0]
    print('Accuracy on Testset:', accuracy)
    truePositive,trueNegative,falsePositive,falseNegative = 0,0,0,0
    for i in range(len(yPred)):
        if np.argmax(yTesting[i,:])==0:
            if yPred[i]==0:
                truePositive+=1
            else:
                falseNegative+=1
        else:
            if yPred[i] == 0:
                falsePositive += 1
            else:
                trueNegative += 1
    print(truePositive,trueNegative,falsePositive,falseNegative)
    totalScores = sorted(decisionValues)
    index = sorted(range(len(decisionValues)), key=decisionValues.__getitem__)
    labels = np.zeros(len(yTesting))
    for i in range(len(labels)):
        labels[i] = np.argmax(yTesting[index[i]])

    truePositive = np.zeros(len(labels) + 1)
    falsePositive = np.zeros(len(labels) + 1)
    for i in range(len(totalScores)):
        if labels[i] < 0.5:
            truePositive[0] += 1
        else:
            falsePositive[0] += 1
    for i in range(len(totalScores)):
        if labels[i] < 0.5:
            truePositive[i + 1] = truePositive[i] - 1
            falsePositive[i + 1] = falsePositive[i]
        else:
            falsePositive[i + 1] = falsePositive[i] - 1
            truePositive[i + 1] = truePositive[i]
    truePositive = truePositive / truePositive[0]
    falsePositive = falsePositive / falsePositive[0]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(falsePositive, truePositive)
    plt.show()
    plt.savefig('ROC_Ours.png')


