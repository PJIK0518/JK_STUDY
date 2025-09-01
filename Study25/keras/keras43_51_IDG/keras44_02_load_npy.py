# 44_1.copy
# https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition

##########################################################################
#0. 준비
##########################################################################
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow_addons.metrics import F1Score

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import time

path_NP = 'C:/Study25/_data/_save_npy/'

# np.save(path_NP + 'keras44_01_x_train.npy', arr = x)
# np.save(path_NP + 'keras44_01_y_train.npy', arr = y)

S = time.time()
x_trn = np.load(path_NP + 'keras44_01_x_train.npy')
y_trn = np.load(path_NP + 'keras44_01_y_train.npy')
E = time.time()

print('로드시간 :',E-S)

print(x_trn)
print(y_trn[:19])

print(x_trn.shape,y_trn.shape)