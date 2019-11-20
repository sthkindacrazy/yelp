import os
import numpy as np
import pandas as pd
import cleaning as cl
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split


basedir = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(basedir, 'data')


# load train
def load_raw_data(dt='train'):
    data = pd.read_csv(data_path + '/Yelp_' + dt + '.csv')
    x_data = data.iloc[:, 1:]
    y_data = data.iloc[:, 0]  # stars
    return x_data, y_data


def load_clean_row(dt='train'):
    data = pd.read_csv(data_path + '/Yelp_' + dt + '.csv')
    data = cl.cleaning(data)
    return data


def load_clean_data(dt='train'):
    data = pd.read_csv(data_path + '/Yelp_' + dt + '.csv')
    data = cl.cleaning(data)
    x_data = data.iloc[:, 1:]
    y_data = data.iloc[:, 0]  # stars
    return x_data, y_data


class Dataset(Dataset):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        self.x = torch.Tensor(self.data[index])
        self.y = self.labels[index]
        return self.x, self.y


def collate(seq_list):
    inputs, targets = zip(*seq_list)
    inputs_lens = torch.IntTensor([len(seq) for seq in inputs])
    inputs = rnn.pad_sequence(inputs, batch_first=True).type(torch.LongTensor)
    targets = torch.LongTensor([label for label in targets])
    return inputs, targets, inputs_lens