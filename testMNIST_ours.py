from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from NN import *
from nn_train import nn_train
from nn_test import nn_test
import numpy as np
import os
import matplotlib.image as mpimg
import pickle
from nn_forward import nn_forward
from nn_predict import nn_predict
from nn_backward import nn_backward
from nn_applygradient import nn_applygradient
from function import sigmoid, softmax
# for log
import logging
import time
from datetime import datetime

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

# ==================== Logging ====================

os.makedirs("logs", exist_ok=True)

run_time = datetime.now().strftime("%Y%m%d_%H%M%S")

train_log_file = f"logs/train_{run_time}.log"
test_log_file = f"logs/test_{run_time}.log"


def create_logger(name, log_file):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # 防止重复添加 handler
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 写入文件
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 输出到终端
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


train_logger = create_logger("train_logger", train_log_file)
test_logger = create_logger("test_logger", test_log_file)

#Training
dir_path = '/home/wzc/nas/home/Dataset/MNIST/train'
file_ls = os.listdir(dir_path)
data = np.zeros((60000, 784), dtype=float)
label = np.zeros((60000, 10), dtype=float)
flag = 0
for dir in file_ls:
    files = os.listdir(dir_path+'/'+dir)
    for file in files:
        filename = dir_path+'/'+dir+'/'+file
        img = mpimg.imread(filename)
        data[flag,:] = np.reshape(img, -1)/255
        label[flag,int(dir)] = 1.0
        flag+=1

ratioTraining = 0.95
xTraining, xValidation, yTraining, yValidation = train_test_split(data, label, test_size=1 - ratioTraining, random_state=0)  # 随机分配数据集


if os.path.exists('storedNN.npz'):
    nn = load_variable('storedNN.npz')
else:
    nn = NN(
    layer=[784, 400, 169, 49, 10],
    batch_normalization=1,
    active_function='relu',
    batch_size=50,
    learning_rate=0.003,
    optimization_method='RMSPropNesterov',
    objective_function='Cross Entropy',
    rho=0.95,
    alpha=0.9,
    stability_constant=1e-8
)


epoch = 0
maxAccuracy = 0
totalAccuracy = []
totalCost = []
maxEpoch = 100

train_logger.info("=" * 60)
train_logger.info("Training started")
train_logger.info("Training samples: %d", len(xTraining))
train_logger.info("Validation samples: %d", len(xValidation))
train_logger.info("Max epochs: %d", maxEpoch)

# 记录网络的一些重要参数
train_logger.info("=" * 60)

training_start_time = time.time()

while epoch < maxEpoch:

    epoch += 1
    epoch_start_time = time.time()

    # -------- Training --------
    nn = nn_train(nn, xTraining, yTraining)

    cost = sum(nn.cost.values()) / len(nn.cost.values())
    totalCost.append(cost)

    # -------- Validation --------
    wrongs, predictedLabel, accuracy,_ = nn_test(
        nn,
        xValidation,
        yValidation
    )

    totalAccuracy.append(accuracy)

    # -------- Save best model --------
    is_best = False

    if accuracy > maxAccuracy:
        maxAccuracy = accuracy
        storedNN = nn
        save_variable(nn, 'storedNN.npz')
        is_best = True

    epoch_time = time.time() - epoch_start_time

    # -------- Logging --------
    train_logger.info(
        "Epoch %03d/%03d | Cost: %.6f | "
        "Validation Accuracy: %.4f | "
        "Time: %.2fs%s",
        epoch,
        maxEpoch,
        cost,
        accuracy,
        epoch_time,
        " | Best model saved" if is_best else ""
    )

training_time = time.time() - training_start_time

train_logger.info("=" * 60)
train_logger.info("Training finished")
train_logger.info("Best Validation Accuracy: %.4f", maxAccuracy)
train_logger.info("Total Training Time: %.2fs", training_time)
train_logger.info("=" * 60)

#Testing
dir_path = '/home/wzc/nas/home/Dataset/MNIST/test'

file_ls = os.listdir(dir_path)

xTesting = np.zeros((10000, 784), dtype=float)
yTesting = np.zeros((10000, 10), dtype=float)

flag = 0

for dir in file_ls:

    class_dir = os.path.join(dir_path, dir)

    # 跳过 .DS_Store 等非目录文件
    if not os.path.isdir(class_dir):
        continue

    files = os.listdir(class_dir)

    for file in files:

        filename = os.path.join(class_dir, file)

        # 防止类别文件夹中也存在 .DS_Store 等文件
        if not os.path.isfile(filename):
            continue

        img = mpimg.imread(filename)

        xTesting[flag, :] = np.reshape(img, -1) / 255
        yTesting[flag, int(dir)] = 1.0

        flag += 1

print("Loaded test images:", flag)

if os.path.exists('storedNN.npz'):

    test_logger.info("=" * 60)
    test_logger.info("Testing started")
    test_logger.info("Test samples: %d", len(xTesting))

    test_start_time = time.time()

    storedNN = load_variable('storedNN.npz')

    wrongs, predictedLabel, accuracy,_ = nn_test(
        storedNN,
        xTesting,
        yTesting
    )

    test_time = time.time() - test_start_time

    test_logger.info("Test Accuracy: %.4f", accuracy)
    test_logger.info("Number of wrong predictions: %d", len(wrongs))
    test_logger.info("Test Time: %.2fs", test_time)

    # Confusion Matrix
    confusionMatrix = np.zeros((10, 10), dtype=int)

    for i in range(len(predictedLabel)):
        trueLabel = np.argmax(yTesting[i, :])
        confusionMatrix[trueLabel, predictedLabel[i]] += 1

    test_logger.info(
        "Confusion Matrix:\n%s",
        np.array2string(confusionMatrix)
    )

    test_logger.info("=" * 60)
    test_logger.info("Testing finished")
    test_logger.info("=" * 60)

else:
    test_logger.error(
        "storedNN.npz does not exist. Testing cannot be performed."
    )