from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Dropout, BatchNormalization

import numpy as np

##########################################################################
#1. 데이터
##########################################################################
path_NP = 'C:/Study25/_data/image/brain/'

x_trn = np.load(path_NP + 'x_trn.npy')
y_trn = np.load(path_NP + 'y_trn.npy')
x_tst = np.load(path_NP + 'x_tst.npy')

# print(x_trn.shape) (160, 150, 150, 3)
# print(y_trn.shape) (160,)
# print(x_tst.shape) (120, 150, 150, 3)
