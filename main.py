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
from sklearn.model_selection import train_test_split



train_data = dl.load_clean_data('train')
x_train, y_train = (train_data[0], train_data[-1])
test_data = dl.load_clean_data('test')
x_test, y_test = (test_data[0], test_data[-1])


def multinomial_bayes():
    x_train_ref_org, x_test_ref_org, y_train_ref, y_test_ref = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    num_feature = []
    acc = []
    for i in range(1000, 10000, 1000):
        mnb = MultinomialNB()
        x_train_ref = x_train_ref_org
        x_test_ref = x_test_ref_org
        #x_train_ref = cl.phrase_tf_idf_encode(x_train_ref, i)
        #x_test_ref = cl.phrase_tf_idf_encode(x_test_ref, i)
        x_train_ref = cl.phrase_one_hot_encode(x_train_ref, i)
        x_test_ref = cl.phrase_one_hot_encode(x_test_ref, i)
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






