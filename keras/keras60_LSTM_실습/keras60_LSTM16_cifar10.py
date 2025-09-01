### 39_3.copy

### [실습]
#1. 시간 : vs CNN, CPU vs GPU
#2. 성능 : 기존 모델 능가

from tensorflow.keras.datasets import cifar10

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
(x_trn, y_trn), (x_tst,y_tst) = cifar10.load_data()

#####################################
### y OneHot
One = OneHotEncoder(sparse=False)

y_trn = y_trn.reshape(-1,1)
y_tst = y_tst.reshape(-1,1)

One.fit(y_trn)

y_trn = One.transform(y_trn)
y_tst = One.transform(y_tst)

# print(x_trn.shape, y_trn.shape) (50000, 32, 32, 3) (50000, 10)
# print(x_tst.shape, y_tst.shape) (10000, 32, 32, 3) (10000, 10)

#####################################
### Scaling
x_trn = (x_trn-127.5)/(510.0)
x_tst = (x_tst-127.5)/(510.0)

#####################################
### Reshape

x_trn = x_trn.reshape(x_trn.shape[0],32*3,32)
x_tst = x_tst.reshape(x_tst.shape[0],32*3,32)

##########################################################################
#2. 모델 구성
##########################################################################
### 모델 불러오기
# path_S = './_save/keras40/cifar10/'
# model = load_model(path_S + 'k39_0611_1803_0002-0.6601.h5')
#####################################
from tensorflow.keras.layers import LSTM

model = Sequential()
model.add(LSTM(10, input_shape = (32*3,32), activation = 'relu'))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(BatchNormalization())
model.add(Dense(32, activation = 'relu'))
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(Dense(16, activation = 'relu'))
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(Dense(10, activation = 'softmax'))

'''
[CNN _ GPU]
loss : 0.6803414821624756
acc  : 0.7702000141143799
acc  : 0.7702

[DNN _ CPU]
loss : 1.3164840936660767
acc  : 0.5637999773025513
acc  : 0.5638
시간 : 541.0520420074463

[DNN _ GPU]
loss : 1.328972339630127
acc  : 0.5616999864578247
acc  : 0.5617
시간 : 729.8132548332214

[LSTM]
loss : 2.2153496742248535
acc  : 0.17159999907016754
acc  : 0.1716
시간 : 72.57322764396667
'''

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

path = './_save/keras40/cifar10/'
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

#####################################
### 가중치 불러오기
# path_W = './_save/keras40/cifar10/'
# model.load_weights(path_W + 'k39_0611_1741_0045-0.6528.h5')

S = time.time()

hist = model.fit(x_trn, y_trn,
                 epochs = 100,
                 batch_size = 10000,
                 verbose = 3,
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
y_tst = np.argmax(y_tst, axis=1)
ACC = accuracy_score(results, y_tst)

print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print("loss :", loss[0])
print("acc  :", loss[1])
print("acc  :", ACC)
print("시간 :", T)