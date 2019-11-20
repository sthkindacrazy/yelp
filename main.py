import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import data_loader as dl
import cleaning as cl
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt


train_data = dl.load_clean_data('train')
x_train, y_train = (train_data[0], train_data[-1])
test_data = dl.load_clean_data('test')
x_test, y_test = (test_data[0], test_data[-1])


def multinomial_bayes():
    masterDataset = dl.Dataset(x_train, y_train)
    val_len = int(len(masterDataset) * 0.2)
    train_len = len(masterDataset) - val_len
    trainDataset, valDataset = random_split(masterDataset, (train_len, val_len))
    #x_train_ref, y_train_ref = (trainDataset.dataset.data, trainDataset.dataset.labels)
    #x_test_ref, y_test_ref = (valDataset.dataset.data, valDataset.dataset.labels)
    num_feature = []
    acc = []
    for i in range(50000, 5000, -5000):
        mnb = MultinomialNB()
        x_train_ref, y_train_ref = (trainDataset.dataset.data, trainDataset.dataset.labels)
        x_test_ref, y_test_ref = (valDataset.dataset.data, valDataset.dataset.labels)
        x_train_ref = cl.phrase_one_hot_encode(x_train_ref, num = i)
        x_test_ref = cl.phrase_one_hot_encode(x_test_ref, num = i)
        mnb.fit(x_train_ref, y_train_ref)
        predmnb = mnb.predict(x_test_ref)
        score = round(accuracy_score(y_test_ref, predmnb) * 100, 2)
        print(i, score)
        num_feature.append(i)
        acc.append(score)
    plt.plot(num_feature, acc)
    plt.xlabel('number of words')
    plt.ylabel('classification accuracy')
    plt.show()






