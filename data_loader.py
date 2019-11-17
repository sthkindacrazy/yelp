import os
import numpy as np
import pandas as pd
import cleaning as cl


basedir = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(basedir, 'data')


# load train
def load_raw_data(dt='train'):
    data = pd.read_csv(data_path + '/Yelp_' + dt + '.csv')
    x_data = data.iloc[:, 1:]
    y_data = data.iloc[:, 0]  # stars
    return x_data, y_data


def load_clean_data(dt='train'):
    data = pd.read_csv(data_path + '/Yelp_' + dt + '.csv')
    data = cl.cleaning(data)
    x_data = data.iloc[:, 1:]
    y_data = data.iloc[:, 0]  # stars
    return x_data, y_data
