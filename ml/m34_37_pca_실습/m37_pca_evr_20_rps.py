# 증폭 : 50-7 복사

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPool2D, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.datasets import fashion_mnist

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import time

##########################################################################
#1. 데이터
##########################################################################
path_NP = 'C:/Study25/_data/tensor_cert/rps/'

x = np.load(path_NP + 'x_trn.npy')
y = np.load(path_NP + 'y_trn.npy')
y = np.argmax(y, axis=1)
# print(x.shape)
# print(y.shape)
# print(np.unique(y))

x_trn,x_tst,y_trn,y_tst = train_test_split(x, y,
                                           train_size=0.7,
                                           shuffle=True,
                                           random_state=42)

### 이미지 데이터 증폭
IDG = ImageDataGenerator(
    # rescale=1./255.,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=15,
    zoom_range=0.2,
    shear_range=0.7,
    fill_mode='nearest' 
)
Augment_size = 10000

x_trn = x_trn/255.
x_tst = x_tst/255.

randidx = np.random.randint(x_trn.shape[0], size = Augment_size)  

x_augmented = x_trn[randidx].copy()

y_augmented = y_trn[randidx].copy()

x_augmented = x_augmented.reshape(x_augmented.shape[0],
                                  x_augmented.shape[1],
                                  x_augmented.shape[2],3)

x_augmented, y_augmented = IDG.flow(
    x = x_augmented,
    y = y_augmented,
    batch_size=Augment_size,
    shuffle=False
).next()

x_trn = x_trn.reshape(-1,150,150,3)
x_tst = x_tst.reshape(-1,150,150,3)

x_trn = np.concatenate([x_trn, x_augmented])
y_trn = np.concatenate([y_trn, y_augmented])

from sklearn.decomposition import PCA
import time

pca = PCA(n_components=x_trn.shape[1])

x_trn = pca.fit_transform(x_trn)
x_tst = pca.fit_transform(x_tst)

evr = pca.explained_variance_ratio_
evr_cumsum = np.cumsum(evr)

a = len(np.where(evr_cumsum >= 1.)[0]) + 1
b = len(np.where(evr_cumsum >= 0.999)[0]) + 1
c = len(np.where(evr_cumsum >= 0.99)[0]) + 1
d = len(np.where(evr_cumsum >= 0.95)[0]) + 1

num = [a,b,c,d]
acc = []

for p in num :
    pca = PCA(n_components=p)
    pca.fit(x_trn)
    x_trn_P = pca.transform(x_trn)
    x_tst_P = pca.transform(x_tst)
    
    S = time.time()

    #2. 모델
    from catboost import CatBoostRegressor, CatBoostClassifier

    model = CatBoostClassifier(verbose=0)
    
    #3. 컴파일 훈련
    model.fit(x_trn_P, y_trn)
    
    score = model.score(x_tst_P, y_tst)
    
    acc.append((p, score))

for p, ACC in acc:
    print(f"n_components={p:>3} | acc={float(ACC):.4f}")

'''
[Conv2D]
ACC  : 1.0

[LSTM]
ACC  : 0.43739837408065796

[Conv1D]
ACC  : 0.39674797654151917

'''