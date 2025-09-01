# 36_5.copy

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPool2D
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

# x를 reshape 
x_trn = x_trn.reshape(60000,28,28,1)
x_tst = x_tst.reshape(10000,28,28,1)
x_tst = x_tst.reshape(x_tst.shape[0], x_tst.shape[1], x_tst.shape[2], 1)
                                            # 이렇게 입력해도 똑같이 인식함
# print(x_trn.shape, x_tst.shape)             (60000, 28, 28, 1) (10000, 28, 28, 1)

y_trn = pd.get_dummies(y_trn)
y_tst = pd.get_dummies(y_tst)
# print(y_trn.shape, y_tst.shape) (60000, 10) (10000, 10)

# x_trn = x_trn.reshape(60000,784)
# x_tst = x_tst.reshape(10000,784)

x_trn = (x_trn-127.5)/(510.0)
x_tst = (x_tst-127.5)/(510.0)

# def Scaler(SC, a, b):
#     SC.fit(a)
#     return SC.transform(a), SC.transform(b)

# x_trn, x_tst = Scaler(MinMaxScaler(), x_trn, x_tst)

# x_trn = x_trn.reshape(60000,28,28,1)
# x_tst = x_tst.reshape(10000,28,28,1)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(64,(3,3), strides=1 ,input_shape=(28,28,1), activation='relu'))  # 출력 형태 : 기존 형태 - kernel_size + 1
model.add(MaxPool2D())                                                            # strides / 포복, kernel_size가 움직이는 픽셀의 크기
                                                               # 통상적으로 strides는 kernel_size 보다는 작게 해야한다
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D())              # MaxPool2D, Pool_size = 2(Default)
                                    # > Conv2D로 증폭 된 값 중 중요성이 높은 값을 (2, 2)로 압축 (연산량 감소, 학습속도 가속)
                                    # > 모델의 성능은 올라가겠지만, 데이터 소실이 있을 수 있어서 통상적으로 Conv2D랑 적절히 배합해서 사용
model.add(Dropout(0.1))

model.add(BatchNormalization())
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(Dropout(0.1))

model.add(Flatten())                                           

model.add(BatchNormalization())
model.add(Dense(units=16, activation='relu'))                
model.add(Dropout(0.1))

model.add(BatchNormalization())
model.add(Dense(units=16, input_shape=(16,), activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(units=10, activation='softmax'))

""" model.summary() : MaxPool2D
 Layer (type)                 Output Shape              Param #   
=================================================================
 conv2d (Conv2D)              (None, 26, 26, 64)        640       
 batch_normalization (BatchN  (None, 26, 26, 64)        256       
 ormalization)
 conv2d_1 (Conv2D)            (None, 24, 24, 64)        36928     
 max_pooling2d (MaxPooling2D  (None, 12, 12, 64)        0
 )
 dropout (Dropout)            (None, 12, 12, 64)        0
 batch_normalization_1 (Batc  (None, 12, 12, 64)        256       
 hNormalization)
 conv2d_2 (Conv2D)            (None, 10, 10, 32)        18464     
 dropout_1 (Dropout)          (None, 10, 10, 32)        0
 flatten (Flatten)            (None, 3200)              0
 batch_normalization_2 (Batc  (None, 3200)              12800
 hNormalization)
 dense (Dense)                (None, 16)                51216
 dropout_2 (Dropout)          (None, 16)                0
 batch_normalization_3 (Batc  (None, 16)                64
 hNormalization) 
 dense_1 (Dense)              (None, 16)                272
 dropout_3 (Dropout)          (None, 16)                0
 dense_2 (Dense)              (None, 10)                170
=================================================================
Total params: 121,066
Trainable params: 114,378
Non-trainable params: 6,688
"""

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
[ MinMax ]
loss : 0.05730954557657242
acc  : 0.9851999878883362
acc  : 0.9852
시간 : 374.8570990562439
        
[ x = x / 255. ]
loss : 0.04574809968471527
acc  : 0.9865999817848206
acc  : 0.9866
시간 : 383.5334916114807 
        
[ x = x / 127.5 ]
loss : 0.04537307843565941
acc  : 0.9865999817848206
acc  : 0.9866
시간 : 366.24327087402344

        [ x = x / 127.5 ] + Batch_N
        loss : 0.0391700305044651
        acc  : 0.9904999732971191
        acc  : 0.9905
        시간 : 690.4066429138184
        
        [ x = x / 255. ] + Batch_N
        loss : 0.03654203936457634
        acc  : 0.9904999732971191
        acc  : 0.9905
        시간 : 591.6692810058594
        
        [ x = (x - 127.5) / 255. ] + Batch_N          
        loss : 0.036810390651226044
        acc  : 0.9907000064849854
        acc  : 0.9907
        시간 : 905.5825564861298
        
        [ x = (x - 127.5) / 510. ] + Batch_N    ############################
        loss : 0.03612019494175911
        acc  : 0.9908999800682068
        acc  : 0.9909
        시간 : 629.7389621734619
        
        loss : 0.026870517060160637
        acc  : 0.9926999807357788
        acc  : 0.9927
        시간 : 613.0614950656891
'''