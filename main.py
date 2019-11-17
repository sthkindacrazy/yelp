import numpy as np
import pandas as pd
import data_loader as dl


train_data = dl.load_clean_data('train')
x_train, y_train = (train_data[0], train_data[-1])
test_data = dl.load_clean_data('test')
x_test, y_test = (test_data[0], test_data[-1])