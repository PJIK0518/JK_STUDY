from tensorflow.keras.datasets import cifar100

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, Model, load_model
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import time

##########################################################################
#1. 데이터
##########################################################################
(x_trn, y_trn), (x_tst,y_tst) = cifar100.load_data()

print(x_trn.shape, y_trn.shape) 
print(x_tst.shape, y_tst.shape)

#####################################
### y OneHot
One = OneHotEncoder(sparse=False)

y_trn = y_trn.reshape(-1,1)
y_tst = y_tst.reshape(-1,1)

One.fit(y_trn)

y_trn = One.transform(y_trn)
y_tst = One.transform(y_tst)

#####################################
### Scaling
x_trn = (x_trn-127.5)/(510.0)
x_tst = (x_tst-127.5)/(510.0)

##########################################################################
#2. 모델 구성
##########################################################################
### 모델 불러오기
path_S = './_save/keras39/cifar100/'
model = load_model(path_S + 'k39_0611_1817_0236-1.8202.h5')

# model = Sequential()
# model.add(Conv2D(100,(3,3), input_shape=(32,32,3), activation='relu',
#                  strides=1, padding='same'))
# model.add(Dropout(0.25))

# model.add(BatchNormalization())
# model.add(Conv2D(filters=80, kernel_size=(3,3), activation='relu',
#                  strides=1, padding='same'))
# model.add(Dropout(0.25))

# model.add(BatchNormalization())
# model.add(Conv2D(filters=60, kernel_size=(3,3), activation='relu',
#                  strides=1, padding='same'))
# model.add(Dropout(0.25))

# model.add(BatchNormalization())
# model.add(Conv2D(filters=40, kernel_size=(3,3), activation='relu',
#                  strides=1, padding='same'))
# model.add(MaxPool2D(pool_size=4))
# model.add(Dropout(0.25))

# model.add(Flatten())                                           

# model.add(BatchNormalization())
# model.add(Dense(units=200, activation='relu'))                  
# model.add(Dropout(0.25))

# model.add(BatchNormalization())
# model.add(Dense(units=150, input_shape=(16,), activation='relu'))
# model.add(Dropout(0.25))

# model.add(Dense(units=100, activation='softmax'))

##########################################################################
#3. 컴파일 훈련
##########################################################################
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['acc']
              )

#####################################
# ### ES
# ES = EarlyStopping(monitor = 'val_loss', mode = 'min',
#                    patience= 50, verbose=1,
#                    restore_best_weights=True,
# )

#####################################
### 파일명
# import datetime

# date = datetime.datetime.now()
# date = date.strftime('%m%d_%H%M')

# path = './_save/keras39/cifar100/'
# filename = '{epoch:04d}-{val_loss:.4f}.h5'
# filepath = "".join([path,'k39_',date, '_', filename])

#####################################
### MCP
# MCP = ModelCheckpoint(monitor = 'val_loss',
#                       mode = 'auto',
#                       save_best_only= True,
#                       verbose=1,
#                       filepath = filepath,
#                       )

# S = time.time()

# hist = model.fit(x_trn, y_trn,
#                  epochs = 5000,
#                  batch_size = 50,
#                  verbose = 1,
#                  validation_split = 0.2,
#                  callbacks = [ES, MCP],
#                  )   

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
loss : 1.812417984008789
acc  : 0.5221999883651733
acc  : 0.5222
시간 : 5364.959238290787
'''



