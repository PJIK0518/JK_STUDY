### 39_4.copy

### [실습]
#1. 시간 : vs CNN, CPU vs GPU
#2. 성능 : 기존 모델 능가

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

#####################################
### reshape
x_trn = x_trn.reshape(x_trn.shape[0],x_trn.shape[1]*x_trn.shape[2]*x_trn.shape[3])
x_tst = x_tst.reshape(x_tst.shape[0],x_tst.shape[1]*x_tst.shape[2]*x_tst.shape[3])

print(x_trn.shape, y_trn.shape) 
print(x_tst.shape, y_tst.shape)

##########################################################################
#2. 모델 구성
##########################################################################
### 모델 불러오기
# path_S = './_save/keras40/cifar10/'
# model = load_model(path_S + 'k39_0611_1803_0002-0.6601.h5')

model = Sequential()
model.add(Dense(521, input_dim = 3072, activation = 'relu'))
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(Dense(100, activation = 'softmax'))

##########################################################################
#3. 컴파일 훈련
##########################################################################
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['acc']
              )

#####################################
### ES
ES = EarlyStopping(monitor = 'val_acc', mode = 'max',
                   patience= 50, verbose=1,
                   restore_best_weights=True,
)

#####################################
### 파일명
import datetime

date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')

path = './_save/keras40/cifar100/'
filename = '{epoch:04d}-{val_loss:.4f}.h5'
filepath = "".join([path,'k40_',date, '_', filename])

#####################################
### MCP
MCP = ModelCheckpoint(monitor = 'val_acc',
                      mode = 'max',
                      save_best_only= True,
                      verbose=1,
                      filepath = filepath,
                      )

S = time.time()

hist = model.fit(x_trn, y_trn,
                 epochs = 5000,
                 batch_size = 50,
                 verbose = 1,
                 validation_split = 0.2,
                 callbacks = [ES, MCP],
                 )   

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
y_tst = np.argmax(y_tst, axis=1)
ACC = accuracy_score(results, y_tst)

print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print("loss :", loss[0])
print("acc  :", loss[1])
print("acc  :", ACC)
print("시간 :", T)

'''
[CNN-GPU]
loss : 1.812417984008789
acc  : 0.5221999883651733
acc  : 0.5222
시간 : 5364.959238290787

[DNN-CPU]
loss : 3.081491708755493
acc  : 0.2833000123500824
acc  : 0.2833
시간 : 577.3938059806824

[DNN-GPU]
loss : 3.1438710689544678
acc  : 0.28769999742507935
acc  : 0.2877
시간 : 701.1971230506897

'''

