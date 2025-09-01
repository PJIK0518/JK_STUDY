# 43_3.copy
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

# from tensorflow.keras import mixed_precision
##########################################################################
#1. 데이터
##########################################################################
# mixed_precision.set_global_policy('mixed_float16')

trn_IDG = ImageDataGenerator(rescale=1./255.)
tst_IDG = ImageDataGenerator(rescale=1./255.)


path_trn = 'C:/Study25/_data/kaggle/cat_dog/train_2/'
S = time.time()
xy_trn = trn_IDG.flow_from_directory(
    path_trn,
    target_size = (32,32),
    batch_size = 100, 
    class_mode = 'binary',
    color_mode = 'rgb',
    shuffle = True,
    seed = 50,
)   # Found 25000 images belonging to 2 classes

# print(xy_trn[0][0].shape) (100, 200, 200, 3)
# print(xy_trn[0][1].shape) (100,)
# print(len(xy_trn)) 250

# exit()

# path_tst = 'C:/Study25/_data/kaggle/cat_dog/test_2/'
# x_tst = tst_IDG.flow_from_directory(
#     path_tst,
#     target_size = (32,32),
#     batch_size = 100,
#     class_mode = 'binary',
#     color_mode = 'rgb',
    # shuffle = True,
    # seed = 50
# )   # Found 12500 images belonging to 1 classes.

# print(x_tst)
# print(len(x_tst))

# plt.imshow(x_tst[0][0][0], 'brg')
# print(x_tst[0][1][0])
# plt.show() 

# exit()

E = time.time()
print('시간 :', E-S)    # 시간 :  0.748157262802124


########## Batch의 list화

all_x = []
all_y = []

for i in range(len(xy_trn)):
    x_batch, y_batch = xy_trn[i]
    all_x.append(x_batch)
    all_y.append(y_batch)
E1 = time.time()

# print(type(all_x)) <class 'list'>
# all_x [[100,200,200,3], [100,200,200,3], [100,200,200,3],... [100,200,200,3]]
# 리스트 안에 numpy가 나열, 그냥 바꿔주면 XXXX
print('시간 :', E1-E)   # 시간 : 39.97046971321106

########## list의 numpy화

x = np.concatenate(all_x, axis=0)
y = np.concatenate(all_y, axis=0)

E2 = time.time()
print('시간 :', E2-E1)   # 시간 : 226.0595293045044

# print(x.shape) (25000, 200, 200, 3) 
# print(y.shape) (25000,)
# print(type(x)) <class 'numpy.ndarray'>
# print(type(y)) <class 'numpy.ndarray'>

path_NP = 'C:/Study25/_data/_save_npy/'

np.save(path_NP + 'keras44_01_x_test_64.npy', arr = x)
np.save(path_NP + 'keras44_01_y_train_0.npy', arr = y)

E3 = time.time()

print('시간 :', E3-E2)  # 시간 : 379.7154459953308


print(xy_trn.class_indices)