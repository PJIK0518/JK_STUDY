### 36_5_모델구성.copy

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.datasets import mnist

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np
import time

##2. 모델 구성
model = Sequential()
model.add(Conv2D(64,(3,3), strides=1 ,input_shape=(28,28,1), activation='relu'))  # 출력 형태 : 기존 형태 - kernel_size + 1
                                                               # strides / 포복, kernel_size가 움직이는 픽셀의 크기
                                                               # 통상적으로 strides는 kernel_size 보다는 작게 해야한다
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(Dropout(0.1))

model.add(BatchNormalization())
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(Dropout(0.1))

model.add(Flatten())                                           # Dense layer와 마찬가지로 Conv2D에서도 activation, Dropout 모두 사용가능

model.add(BatchNormalization())
model.add(Dense(units=16, activation='relu'))                  # 마찬가지로 성능이 향상 될 수 있다
model.add(Dropout(0.1))

model.add(BatchNormalization())
model.add(Dense(units=16, input_shape=(16,), activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(units=10, activation='softmax'))

""" model.summary()
Total params: 366,826
Trainable params: 335,562
Non-trainable params: 31,264 """

# Input_L = Input(shape=(28,28,1))

# conv2_1 = Conv2D(64,(3,3), strides=1, activation='relu')(Input_L)

# batch_2 = BatchNormalization()(conv2_1)
# conv2_2 = Conv2D(64,(3,3), activation='relu')(batch_2)
# dropO_2 = Dropout(0.1)(conv2_2)

# batch_3 = BatchNormalization()(dropO_2)
# conv2_3 = Conv2D(32,(3,3), activation='relu')(batch_3)
# dropO_3 = Dropout(0.1)(conv2_3)

# flatt_4 = Flatten()(dropO_3)

# batch_5 = BatchNormalization()(flatt_4)
# dense_5 = Dense(16, activation='relu')(batch_5)
# dropO_5 = Dropout(0.1)(dense_5)

# batch_6 = BatchNormalization()(dropO_5)
# dense_6 = Dense(16, activation='relu')(batch_6)
# dropO_6 = Dropout(0.1)(dense_6)

# Outpt_L = Dense(10, activation='softmax')(dropO_6)

# model = Model(inputs= Input_L, outputs=Outpt_L)

""" model.summary()
Total params: 366,826
Trainable params: 335,562
Non-trainable params: 31,264 """

exit()
