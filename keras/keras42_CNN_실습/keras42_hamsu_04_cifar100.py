### 39_4_모델구성.copy

from tensorflow.keras.datasets import cifar100

from tensorflow.keras.layers import Dense,Input, Conv2D, Flatten, Dropout, BatchNormalization, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, Model
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import time

##########################################################################
#2. 모델 구성
##########################################################################
model = Sequential()
model.add(Conv2D(100,(3,3), input_shape=(32,32,3), activation='relu',
                 strides=1, padding='same'))
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(filters=80, kernel_size=(3,3), activation='relu',
                 strides=1, padding='same'))
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(filters=60, kernel_size=(3,3), activation='relu',
                 strides=1, padding='same'))
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(filters=40, kernel_size=(3,3), activation='relu',
                 strides=1, padding='same'))
model.add(MaxPool2D(pool_size=4))
model.add(Dropout(0.25))

model.add(Flatten())                                           

model.add(BatchNormalization())
model.add(Dense(units=200, activation='relu'))                  
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Dense(units=150, input_shape=(16,), activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(units=100, activation='softmax'))

##########################################################################
##########################################################################

Input_L = Input(shape=(32,32,3))

cnv2D_1 = Conv2D(100,3, activation='relu',strides=1,padding='same')(Input_L)
dropO_1 = Dropout(0.25)(cnv2D_1)

batch_2 = BatchNormalization()(dropO_1)
cnv2D_2 = Conv2D(80,3, activation='relu',strides=1,padding='same')(batch_2)
dropO_2 = Dropout(0.25)(cnv2D_2)

batch_3 = BatchNormalization()(dropO_2)
cnv2D_3 = Conv2D(60,3, activation='relu',strides=1,padding='same')(batch_3)
dropO_3 = Dropout(0.25)(cnv2D_3)

batch_4 = BatchNormalization()(dropO_3)
cnv2D_4 = Conv2D(40,3, activation='relu',strides=1,padding='same')(batch_4)
maxpl_4 = MaxPool2D(4)(cnv2D_4)
dropO_4 = Dropout(0.25)(maxpl_4)

flatt_5 = Flatten()(dropO_4)

batch_6 = BatchNormalization()(flatt_5)
dense_6 = Dense(200, activation='relu')(batch_6)
dropO_6 = Dropout(0.25)(dense_6)

batch_7 = BatchNormalization()(dropO_6)
dense_7 = Dense(150, activation='relu')(batch_7)
dropO_7 = Dropout(0.25)(dense_7)

Outpt_L = Dense(100, activation='softmax')(dropO_7)

model = Model(inputs = Input_L, outputs = Outpt_L)

model.summary()
# Total params: 709,230
# Trainable params: 703,230
# Non-trainable params: 6,000