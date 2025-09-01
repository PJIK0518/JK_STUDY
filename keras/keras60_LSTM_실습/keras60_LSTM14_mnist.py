
### 40_5.copy

import numpy as np
import pandas as pd
import time

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
##########################################################################
#1. 데이터
##########################################################################
(x_trn, y_trn), (x_tst, y_tst) = mnist.load_data()

""" print(x_trn.shape, y_trn.shape) (60000, 28, 28) (60000,)"""
""" print(x_tst.shape, y_tst.shape) (10000, 28, 28) (10000,)"""

#####################################
### Scaling
x_trn = x_trn/255.0
x_tst = x_tst/255.0

""" print(np.max(x_trn), np.min(x_trn)) 1.0 / 0.0 """
""" print(np.max(x_tst), np.min(x_tst)) 1.0 / 0.0 """

x_trn = x_trn.reshape(x_trn.shape[0], x_trn.shape[1]*x_trn.shape[2])
x_tst = x_tst.reshape(x_tst.shape[0], x_tst.shape[1]*x_tst.shape[2])

""" print(x_trn.shape) (60000, 784)"""
""" print(x_tst.shape) (10000, 784)"""

#####################################
### OneHot
One = OneHotEncoder(sparse=False)

# sklearn은 matrix 형태의 데이터를 요구
# but. 지금 y_trn : (60000,) / y_tst : (10000,) : vector

y_trn = y_trn.reshape(60000, 1)
y_tst = y_tst.reshape(-1, 1) # -1 : 데이터의 가장 끝을 의미
                             # -10000 : 10000번째 데이터...??
                             
""" print(y_trn.shape) (60000, 1) """
""" print(y_tst.shape) (10000, 1) """

y_trn = One.fit_transform(y_trn)
y_tst = One.fit_transform(y_tst)

""" print(y_trn.shape) (60000, 10) """
""" print(y_tst.shape) (10000, 10) """

##########################################################################
#2. 모델 구성
##########################################################################
### [실습]
#1. 시간체크 : vs CNN, CPU vs GPU
#2. 성능 0.98 ACC
#####################################
### 불러오기
path = './_save/keras40/mnist/'
# model = load_model(path + 'k36_0612_1117_0032-0.0693.h5')
x_trn = np.array(x_trn).reshape(-1,49,16)
x_tst = np.array(x_tst).reshape(-1,49,16)

from tensorflow.python.keras.layers import LSTM


model = Sequential()
model.add(LSTM(15, input_shape=(49,16), activation = 'relu'))
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(Dense(32, activation = 'relu'))
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(Dense(10, activation = 'softmax'))

""" 성능 비교
[CNN]
[ x = x / 255. ] + Batch_N
loss : 0.03654203936457634
acc  : 0.9904999732971191
acc  : 0.9905
시간 : 591.6692810058594

[DNN-GPU]
loss : 0.06617521494626999
acc  : 0.9856

[DNN-GPU]
loss : 0.07521674036979675
acc  : 0.9867
시간 : 2019.2730367183685

[DNN-CPU]
loss : 0.07164125889539719
acc  : 0.9871
시간 : 1211.6511850357056

[LSTM-CPU]
loss : 0.09292933344841003
ACC  : 0.9722222089767456
F1sc : 0.9762872822934563
time : 2.6603565216064453
"""

#3. 컴파일 훈련
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['acc']
              )

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

ES = EarlyStopping(monitor = 'val_acc', mode = 'max',
                   patience= 50, verbose=1,                         # ES에서의 verbose = early stopping 지점을 알 수 있다
                   restore_best_weights=True,
    
)
################################# mpc 세이브 파일명 만들기 #################################
### 월일시 넣기
import datetime

date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')

filename = '{epoch:04d}-{val_loss:.4f}.h5' 
filepath = "".join([path,'k40_',date, '_', filename])

MCP = ModelCheckpoint(monitor='val_acc',
                      mode='max',
                      save_best_only=True,
                      verbose = 1,
                      filepath= filepath 
                      )

Start = time.time()
hist = model.fit(x_trn, y_trn,
                 epochs = 5000,
                 batch_size = 50,        
                 verbose = 3,
                 validation_split = 0.2,
                 callbacks = [ES, MCP])                   
End = time.time()

T = End - Start

#4. 평가,예측
loss = model.evaluate(x_tst, y_tst,
                      verbose= 1)
results = model.predict(x_tst)
results = np.argmax(results, axis=1)
y_tst = np.argmax(y_tst, axis=1)
ACC = accuracy_score(results, y_tst)

print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print("loss :", loss[0])
print("acc  :", ACC)
print("시간 :", T)
