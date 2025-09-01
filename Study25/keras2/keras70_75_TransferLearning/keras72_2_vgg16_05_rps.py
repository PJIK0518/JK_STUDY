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
path_NP = './Study25/_data/tensor_cert/rps/'

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

##########################################################################
#2. 모델 구성
##########################################################################
### 랜덤고정
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
import tensorflow as tf
import numpy as np
import random
RS = 111
random.seed(RS)
np.random.seed(RS)
tf.random.set_seed(RS)


from tensorflow.keras.applications import VGG16

vgg16 = VGG16(include_top = False,
              input_shape = (150,150,3))

# vgg16.trainable = False

model = Sequential()

model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10, activation = 'softmax'))

### 실습
# [FLATTEN]
# 1. 전이학습 vs 기존최고기록 (acc  : 0.7702)
# 2. vgg 16 가중치 동결
        # loss : 0.00011754405568353832
        # ACC  : 1.0
# 3. vgg 16 가중치 안동결

# [GAP]
# 1. 전이학습 vs 기존최고기록 (acc  : 0.7702)
# 2. vgg 16 가중치 동결

# 3. vgg 16 가중치 안동결

'''
[Conv2D]
ACC  : 1.0

[LSTM]
ACC  : 0.43739837408065796

'''

##########################################################################
#3. 컴파일 훈련
##########################################################################
model.compile(
    loss = 'sparse_categorical_crossentropy',
    optimizer = 'adam',
    metrics=['acc']
)


H = model.fit(
    x_trn, y_trn,
    epochs = 10,
    batch_size = 200,
    verbose = 1,
)

#####################################
### 모델 및 가중치 저장
# model.save(path_S + f'save_{saveNum}.h5')
# model.save_weights(path_W + f'weights_{saveNum}.h5')

##########################################################################
#4. 평가 예측
##########################################################################
LSAC = model.evaluate(x_tst, y_tst)

y_pred = model.predict(x_tst)
y_pred = np.argmax(y_pred, axis=1)  

ACC = accuracy_score(y_tst,y_pred)

print('loss :', LSAC[0])
print('ACC  :', LSAC[1])
# plt.show()