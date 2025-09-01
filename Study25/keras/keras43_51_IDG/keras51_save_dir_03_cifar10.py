# 증폭 : 50-2 복사

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.datasets import cifar10
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import fashion_mnist

import matplotlib.pyplot as plt
import numpy as np

##########################################################################
#1. 데이터
##########################################################################
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
Augment_size = 40000

(x_trn, y_trn), (x_tst, y_tst) = cifar10.load_data()
#####################################
### y OneHot
One = OneHotEncoder(sparse=False)

y_trn = y_trn.reshape(-1,1)
y_tst = y_tst.reshape(-1,1)

One.fit(y_trn)

y_trn = One.transform(y_trn)
y_tst = One.transform(y_tst)

x_trn = (x_trn-127.5)/(510.0)
x_tst = (x_tst-127.5)/(510.0)

randidx = np.random.randint(x_trn.shape[0], size = Augment_size)  

x_augmented = x_trn[randidx].copy()

y_augmented = y_trn[randidx].copy()

x_augmented = x_augmented.reshape(40000,32,32,3)

x_augmented, y_augmented = IDG.flow(
    x = x_augmented,
    y = y_augmented,
    batch_size=Augment_size,
    shuffle=False,
    save_to_dir='c:/Study25/_data/_save_img/03_cifar10/',
).next()

exit()

x_trn = x_trn.reshape(-1,32,32,3)
x_tst = x_tst.reshape(-1,32,32,3)

x_trn = np.concatenate([x_trn, x_augmented])
y_trn = np.concatenate([y_trn, y_augmented])

print(x_trn.shape)
print(y_trn.shape)
print(x_tst.shape)
print(y_tst.shape)

##########################################################################
#2. 모델 구성
##########################################################################
### 모델 불러오기
path_S = './_save/keras39/cifar10/'
model = load_model(path_S + 'k39_0611_1803_0001-0.6651.h5')
#####################################

# model = Sequential()
# model.add(Conv2D(60,(3,3), input_shape=(32,32,3), activation='relu',
#                  strides=1, padding='same'))
# model.add(Dropout(0.25))

# model.add(BatchNormalization())
# model.add(Conv2D(filters=50, kernel_size=(3,3), activation='relu',
#                  strides=1, padding='same'))
# model.add(Dropout(0.25))

# model.add(BatchNormalization())
# model.add(Conv2D(filters=40, kernel_size=(3,3), activation='relu',
#                  strides=1, padding='same'))
# model.add(Dropout(0.25))

# model.add(BatchNormalization())
# model.add(Conv2D(filters=20, kernel_size=(3,3), activation='relu',
#                  strides=1, padding='same'))
# model.add(MaxPool2D())
# model.add(Dropout(0.25))

# model.add(Flatten())                                           

# model.add(BatchNormalization())
# model.add(Dense(units=16, activation='relu'))                  
# model.add(Dropout(0.25))

# model.add(BatchNormalization())
# model.add(Dense(units=16, input_shape=(16,), activation='relu'))
# model.add(Dropout(0.25))

# model.add(Dense(units=10, activation='softmax'))

##########################################################################
#3. 컴파일 훈련
##########################################################################
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['acc']
              )

#####################################
### ES
ES = EarlyStopping(monitor = 'val_loss', mode = 'min',
                   patience= 50, verbose=1,
                   restore_best_weights=True,
)

#####################################
### 파일명
# import datetime

# date = datetime.datetime.now()
# date = date.strftime('%m%d_%H%M')

# path = './_save/keras39/cifar10/'
# filename = '{epoch:04d}-{val_loss:.4f}.h5'
# filepath = "".join([path,'k50_',date, '_', filename])

# #####################################
# ### MCP
# MCP = ModelCheckpoint(monitor = 'val_loss',
#                       mode = 'auto',
#                       save_best_only= True,
#                       verbose=1,
#                       filepath = filepath,
#                       )

#####################################
### 가중치 불러오기
# path_W = './_save/keras39/cifar10/'
# model.load_weights(path_W + 'k39_0611_1741_0045-0.6528.h5')

# S = time.time()

# hist = model.fit(x_trn, y_trn,
#                  epochs = 5000,
#                  batch_size = 50,
#                  verbose = 3,
#                  validation_split = 0.2,
#                  callbacks = [ES, MCP])   

# E = time.time()

# T = E - S

##########################################################################
#4. 평가,예측
##########################################################################
loss = model.evaluate(x_tst, y_tst,
                      verbose= 1)
results = model.predict(x_tst)

#####################################
### 결과값 처리
results = np.argmax(results, axis=1)
y_tst = np.argmax(y_tst, axis=1)
ACC = accuracy_score(results, y_tst)

print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print("loss :", loss[0])
print("acc  :", loss[1])
print("acc  :", ACC)
# print("시간 :", T)

'''
loss : 0.6854364275932312
acc  : 0.7700999975204468
acc  : 0.7701

loss : 0.6854364275932312
acc  : 0.7700999975204468
acc  : 0.7701
'''