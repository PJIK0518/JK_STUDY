###### EVENT ###### 
# 메모리를 터쳐라! #

from tensorflow.keras.datasets import cifar100

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, Model
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import time

##########################################################################
#1. 데이터
##########################################################################
(x_trn, y_trn), (x_tst,y_tst) = cifar100.load_data()

print(x_trn.shape, y_trn.shape) (50000, 32, 32, 3) (50000, 1)
print(x_tst.shape, y_tst.shape)

#####################################
### y OneHot
One = OneHotEncoder(sparse=False)

y_trn = y_trn.reshape(-1,1)
y_tst = y_tst.reshape(-1,1)

One.fit(y_trn)

y_trn = One.transform(y_trn)
y_tst = One.transform(y_tst)

##########################################################################
#2. 모델 구성
##########################################################################
model = Sequential()
model.add(Conv2D(50000,(2,2), input_shape=(28,28,1), activation='relu',
                 strides=1, padding='same'))

model.add(Conv2D(filters=40000, kernel_size=(2,2), activation='relu',
                 strides=1, padding='same'))

model.add(Conv2D(filters=30000, kernel_size=(2,2), activation='relu',
                 strides=1, padding='same'))

model.add(Conv2D(filters=20000, kernel_size=(2,2), activation='relu',
                 strides=1, padding='same'))

model.add(Flatten())                                           

model.add(Dense(units=10000, activation='relu'))                  

model.add(Dense(units=5000, input_shape=(16,), activation='relu'))

model.add(Dense(units=10, activation='softmax'))

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
import datetime

date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')

path = './_save/keras39/cifar100/'
filename = '{epoch:04d}-{val_loss:.4f}.h5'
filepath = "".join([path,'k39_',date, '_', filename])

#####################################
### MCP
MCP = ModelCheckpoint(monitor = 'val_loss',
                      mode = 'auto',
                      save_best_only= True,
                      verbose=1,
                      filepath = filepath,
                      )

S = time.time()

hist = model.fit(x_trn, y_trn,
                 epochs = 100000,
                 batch_size = 1,
                 verbose = 1,
                 validation_split = 0.2,
                 callbacks = [ES, MCP])   

E = time.time()

T = E - S

##########################################################################
#4. 평가,예측
##########################################################################
loss = model.evaluate(x_tst, y_tst,
                      verbose= 1)
results = model.predict(x_tst)

#####################################
### 결과값 처리
results = np.argmax(results, axis=1)
y_tst = np.argmax(y_tst.values, axis=1)
ACC = accuracy_score(results, y_tst)

print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print("loss :", loss[0])
print("acc  :", loss[1])
print("acc  :", ACC)
print("시간 :", T)