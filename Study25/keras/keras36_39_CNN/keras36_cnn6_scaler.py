# 36_5.copy

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.datasets import mnist

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np
import time

#1. 데이터
(x_trn, y_trn), (x_tst, y_tst) = mnist.load_data()

# print(x_trn.shape, y_trn.shape)             (60000, 28, 28) (60000,)
# print(np.unique(y_trn, return_counts=True))  array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8
#                                              array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],dtype=int64

### Scaling 1. MinMaxScaler
x_trn = x_trn.reshape(60000,28*28)                                      # 연산만 던져줘도 가능
x_tst = x_tst.reshape(x_tst.shape[0], x_tst.shape[1]*x_tst.shape[2])    # shape column 별로 적용해도 가능
print(x_trn.shape, x_tst.shape)     # (60000, 784) (10000, 784)

scaler = MinMaxScaler()
scaler.fit(x_trn)                   # ERROR : ValueError: Found array with dim 3. MinMaxScaler expected <= 2.
                                            # 2차원 이하의 데이터여야하는데 우리 데이터는 3차원이다...
                                            # >> sklearn의 Scaler은 2차원 이하의 데이터만 호환된다!
x_trn = scaler.transform(x_trn)     #  1.0, 0.0
x_tst = scaler.transform(x_tst)     # 24.0, 0.0 >> 784의 column 안에서 각각의 최소 최대값이 달라서 발생 가능한 문제
                                    # 즉, 모든 column이 같은 기준으로 변환되지 않기 때문에, 통상적으로 사용하지 않음

"""
x_trn = x_trn.reshape(60000,784)
x_tst = x_tst.reshape(10000,784)

def Scaler(SC, a, b):
    SC.fit(a)
    return SC.transform(a), SC.transform(b)

x_trn, x_tst = Scaler(MinMaxScaler(), x_trn, x_tst)

x_trn = x_trn.reshape(60000,28,28,1)
x_tst = x_tst.reshape(10000,28,28,1)
"""

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


# print(x_trn.shape, x_tst.shape)
# print(np.max(x_trn), np.min(x_trn))
# print(np.max(x_tst), np.min(x_tst))
# exit()

# x를 reshape (to(60000,28,28,1)) 
x_trn = x_trn.reshape(60000,28,28,1)
x_tst = x_tst.reshape(10000,28,28,1)
x_tst = x_tst.reshape(x_tst.shape[0], x_tst.shape[1], x_tst.shape[2], 1)
                                            # 이렇게 입력해도 똑같이 인식함
# print(x_trn.shape, x_tst.shape)             (60000, 28, 28, 1) (10000, 28, 28, 1)


#################################
### Scaling
y_trn = pd.get_dummies(y_trn)
y_tst = pd.get_dummies(y_tst)
# print(y_trn.shape, y_tst.shape) (60000, 10) (10000, 10)


#2. 모델 구성
model = Sequential()
model.add(Conv2D(64,(3,3), strides=1 ,input_shape=(28,28,1), activation='relu'))  # 출력 형태 : 기존 형태 - kernel_size + 1
                                                               # strides / 포복, kernel_size가 움직이는 픽셀의 크기
                                                               # 통상적으로 strides는 kernel_size 보다는 작게 해야한다
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(Dropout(0.1))
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(Dropout(0.1))
model.add(Flatten())                                           # Dense layer와 마찬가지로 Conv2D에서도 activation, Dropout 모두 사용가능
model.add(Dense(units=16, activation='relu'))                  # 마찬가지로 성능이 향상 될 수 있다
model.add(Dropout(0.1))
model.add(Dense(units=16, input_shape=(16,), activation='relu'))
model.add(Dense(units=10, activation='softmax'))

""" model.summary() : 모델 수정 후
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 13, 13, 64)        640
 conv2d_1 (Conv2D)           (None, 11, 11, 64)        36928
 dropout (Dropout)           (None, 11, 11, 64)        0
 conv2d_2 (Conv2D)           (None, 9, 9, 32)          18464
 flatten (Flatten)           (None, 2592)              0
 dense (Dense)               (None, 16)                41488
 dropout_1 (Dropout)         (None, 16)                0
 dense_1 (Dense)             (None, 16)                272
 dense_2 (Dense)             (None, 10)                170
=================================================================
Total params: 97,962
Trainable params: 97,962
Non-trainable params: 0 """
""" model.summary() strides = 1 (Default)
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 26, 26, 64)        640
 conv2d_1 (Conv2D)           (None, 24, 24, 32)        18464
 conv2d_2 (Conv2D)           (None, 22, 22, 16)        4624
 flatten (Flatten)           (None, 7744)              0
 dense (Dense)               (None, 16)                123920
 dense_1 (Dense)             (None, 16)                272
 dense_2 (Dense)             (None, 10)                170
=================================================================
Total params: 148,090
Trainable params: 148,090
Non-trainable params: 0 """
""" model.summary() strides = 2
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 13, 13, 64)        640       
 conv2d_1 (Conv2D)           (None, 11, 11, 32)        18464     
 conv2d_2 (Conv2D)           (None, 9, 9, 16)          4624
 flatten (Flatten)           (None, 1296)              0
 dense (Dense)               (None, 16)                20752
 dense_1 (Dense)             (None, 16)                272
 dense_2 (Dense)             (None, 10)                170
=================================================================
Total params: 44,922
Trainable params: 44,922
Non-trainable params: 0 """

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
                 batch_size = 50,        # CNN에서 batch_size : 그림을 몇 장씩 한번에 처리하냐
                 verbose = 3,
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