# 58_1.copy
## [실습] : LSTM 넣기 ##

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Reshape, LSTM, MaxPool2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.datasets import mnist

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np
import time

#1. 데이터
(x_trn, y_trn), (x_tst, y_tst) = mnist.load_data()

### Scaling 2. 정규화 : 유사 MinMax _ 직접 수기로 진행 / 통상적으로 사용하는 방법
x_trn = x_trn/255.  # (60000, 28, 28) 1.0, 0.0
x_tst = x_tst/255.  # (10000, 28, 28) 1.0, 0.0
                    # 데이터 편향의 문제를 해결 하지 못함

### Scaling 3. 유사 MaxAbs / 통상적으로 사용하는 방법
#   데이터의 중간 지점(127.5)를 0으로 잡고 전체 데이터를 -1~1로 만듬
x_trn = (x_trn - 127.5)/(127.5) # (60000, 28, 28) 1.0, -1.0
x_tst = (x_tst - 127.5)/(127.5) # (10000, 28, 28) 1.0, -1.0
                                # 데이터 편향의 문제를 해결 하지 못함
                                # activation에 따라서 데이터 소멸 가능성 ex) relu, sigmoid
                                # batch_normalization의 경우에는 이론상으로는 -1, 1 사이가 효과적
                                # >> 어찌됐든, 써보고 괜찮은거!


""" reshape를 제거해서 3차원 상태로 유지
# x를 reshape (to(60000,28,28,1)) 
x_trn = x_trn.reshape(60000,28,28,1)
x_tst = x_tst.reshape(10000,28,28,1)
x_tst = x_tst.reshape(x_tst.shape[0], x_tst.shape[1], x_tst.shape[2], 1)
                                            # 이렇게 입력해도 똑같이 인식함
# print(x_trn.shape, x_tst.shape)             (60000, 28, 28, 1) (10000, 28, 28, 1)
"""

#################################
### Scaling
y_trn = pd.get_dummies(y_trn)
y_tst = pd.get_dummies(y_tst)
# print(y_trn.shape, y_tst.shape) (60000, 10) (10000, 10)

''' Conv2D로는 3차원을 못받아드림 '''
#2. 모델 구성
model = Sequential()                        
model.add(Conv2D(10,(4,4), strides=1, input_shape = (28,28,1) ,activation='relu'))
model.add(MaxPool2D(5))
model.add(Dropout(0.1))

model.add(Reshape(target_shape=(25,10)))
model.add(LSTM(50, activation='relu'))
model.add(Dropout(0.1))

model.add(Reshape(target_shape=(5,5,2)))
model.add(Conv2D(10,(4,4), strides=1, activation='relu'))
model.add(Dropout(0.1))

model.add(Reshape(target_shape=(4,10)))
model.add(LSTM(5, activation='relu'))
model.add(Dropout(0.1))

model.add(Flatten())

model.add(Dense(units=10, activation='softmax'))

#3. 컴파일 훈련
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['acc']
              )

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

ES = EarlyStopping(monitor = 'val_loss', mode = 'min',
                   patience= 50, verbose=1,                         # ES에서의 verbose = early stopping 지점을 알 수 있다
                   restore_best_weights=True,
    
)
################################# mpc 세이브 파일명 만들기 #################################
### 월일시 넣기
import datetime

date = datetime.datetime.now()
# print(date)             # 2025-06-02 13:01:01.062409 >> 이 값에서 원하는 부분만 잘라오기
#                                                    # >> 문자열로 변경 필요 (string)
# print(type(date))       # <class 'datetime.datetime'>
date = date.strftime('%m%d_%H%M')              
# % + y : 년
    # m : 월
    # d : 일
    # H : 시
    # M : 분
    # S : 초

# print(date)             # 250602_130556
# print(type(date))       # <class 'str'>

### 모델 정보 넣기 (epoch, weight)
### 파일 저장

path = './_save/keras36_cnn5/'
filename = '{epoch:04d}-{val_loss:.4f}.h5' # 04d : 네자리 수 /// .f4 : 소수넷째자리
filepath = "".join([path,'k36_',date, '_', filename])

MCP = ModelCheckpoint(monitor='val_loss',
                      mode='auto',
                      save_best_only=True,
                      verbose = 1,
                      filepath= filepath # 확장자의 경우 h5랑 같음
                                         # patience 만큼 지나기전 최저 갱신 지점        
                      )
Start = time.time()
hist = model.fit(x_trn, y_trn,
                 epochs = 5000,
                 batch_size = 1000,
                 verbose = 1,
                 validation_split = 0.2,
                 callbacks = [ES, MCP])                    # 두 개 이상을 불러올수도 있다
End = time.time()

T = End - Start

#4. 평가,예측
loss = model.evaluate(x_tst, y_tst,
                      verbose= 1)
results = model.predict(x_tst)
results = np.argmax(results, axis=1)
y_tst = np.argmax(y_tst.values, axis=1)
ACC = accuracy_score(results, y_tst)

print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print("loss :", loss[0])    # loss, categorical_crossentropy
print("acc  :", loss[1])    # metrics, accuracy
print("acc  :", ACC)
print("시간 :", T)

'''
[CPU]
loss : 0.08447165787220001
acc  : 0.9812999963760376
acc  : 0.9813
시간 : 1196.8844635486603

[GPU]
        loss : 0.05730954557657242
        acc  : 0.9851999878883362
        acc  : 0.9852
        시간 : 374.8570990562439 ########################
'''