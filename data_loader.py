import os
import numpy as np
import pandas as pd

basedir = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(basedir, 'data')


# load train
def load_data(dt='train'):
    data = pd.read_csv(data_path + 'Yelp_' + dt)
    x_data = data.iloc[:, 1:]
    y_data = data.iloc[:, 0] # stars
    return x_data, y_data
