# 39_2_모델구성.copy

from tensorflow.keras.layers import Dense,Input, Conv2D, Flatten, Dropout, BatchNormalization, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.datasets import fashion_mnist

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np
import time

##########################################################################
#2. 모델 구성
### 모델 불러오기
path = './_save/keras39/fashion/'
# model = load_model(path + 'k39_0612_0903.h5')
#####################################

# model = Sequential()
# model.add(Conv2D(100, 2, input_shape=(28,28,1), activation='relu',
#                  strides=1, padding='valid'))
# model.add(Dropout(0.2))

# model.add(BatchNormalization())
# model.add(Conv2D(filters=80, kernel_size=2, activation='relu',
#                  strides=1, padding='valid'))
# model.add(Dropout(0.2))

# model.add(BatchNormalization())
# model.add(Conv2D(filters=60, kernel_size=2, activation='relu',
#                  strides=1, padding='valid'))
# model.add(Dropout(0.2))

# model.add(BatchNormalization())
# model.add(Conv2D(filters=40, kernel_size=2, activation='relu',
#                  strides=1, padding='valid'))
# model.add(Dropout(0.2))

# model.add(BatchNormalization())
# model.add(Conv2D(filters=20, kernel_size=2, activation='relu',
#                  strides=1, padding='valid'))
# model.add(MaxPool2D(4))
# model.add(Dropout(0.2))

# model.add(Flatten())                                           

# model.add(BatchNormalization())
# model.add(Dense(units=100, activation='relu'))                  
# model.add(Dropout(0.2))

# model.add(BatchNormalization())
# model.add(Dense(units=80, activation='relu'))                  
# model.add(Dropout(0.2))

# model.add(BatchNormalization())
# model.add(Dense(units=50, activation='relu'))                  
# model.add(Dropout(0.2))

# model.add(BatchNormalization())
# model.add(Dense(units=20, input_shape=(16,), activation='relu'))
# model.add(Dropout(0.2))

# model.add(Dense(units=10, activation='softmax'))
# Total params: 132,200
# Trainable params: 130,180
# Non-trainable params: 2,020
##########################################################################
##########################################################################

Input_L = Input(shape=(28,28,1))

cnv2D_1 = Conv2D(100,2,activation="relu",strides=1,padding='valid')(Input_L)
dropO_1 = Dropout(0.2)(cnv2D_1)

datch_2 = BatchNormalization()(dropO_1)
cnv2D_2 = Conv2D(80,2,activation="relu",strides=1,padding='valid')(datch_2)
dropO_2 = Dropout(0.2)(cnv2D_2)

datch_3 = BatchNormalization()(dropO_2)
cnv2D_3 = Conv2D(60,2,activation="relu",strides=1,padding='valid')(datch_3)
dropO_3 = Dropout(0.2)(cnv2D_3)

datch_4 = BatchNormalization()(dropO_3)
cnv2D_4 = Conv2D(40,2,activation="relu",strides=1,padding='valid')(datch_4)
dropO_4 = Dropout(0.2)(cnv2D_4)

datch_5 = BatchNormalization()(dropO_4)
cnv2D_5 = Conv2D(20,2,activation="relu",strides=1,padding='valid')(datch_5)
maxpl_5 = MaxPool2D(4)(cnv2D_5)
dropO_5 = Dropout(0.2)(maxpl_5)

flatt_6 = Flatten()(dropO_5)

datch_7 = BatchNormalization()(flatt_6)
dense_7 = Dense(100,activation="relu")(datch_7)
dropO_7 = Dropout(0.2)(dense_7)

datch_8 = BatchNormalization()(dropO_7)
dense_8 = Dense(80,activation="relu")(datch_8)
dropO_8 = Dropout(0.2)(dense_8)

datch_9 = BatchNormalization()(dropO_8)
dense_9 = Dense(50,activation="relu")(datch_9)
dropO_9 = Dropout(0.2)(dense_9)

datch_10 = BatchNormalization()(dropO_9)
dense_10 = Dense(20, input_shape=(16,),activation="relu")(datch_10)
dropO_10 = Dropout(0.2)(dense_10)

Outpt_L = Dense(10, activation="softmax")(dropO_10)

M = Model(inputs = Input_L, outputs = Outpt_L)

# M.summary()
# Total params: 132,200
# Trainable params: 130,180
# Non-trainable params: 2,020
