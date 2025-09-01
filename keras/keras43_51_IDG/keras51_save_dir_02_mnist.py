# 증폭 : 50-1 복사

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
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

(x_trn, y_trn), (x_tst, y_tst) = mnist.load_data()

x_trn = x_trn/255.
x_tst = x_tst/255.

randidx = np.random.randint(x_trn.shape[0], size = Augment_size)  

x_augmented = x_trn[randidx].copy()

y_augmented = y_trn[randidx].copy()

x_augmented = x_augmented.reshape(40000,28,28,1)
x_augmented = x_augmented.reshape(x_augmented.shape[0],
                                  x_augmented.shape[1],
                                  x_augmented.shape[2],1)

x_augmented, y_augmented = IDG.flow(
    x = x_augmented,
    y = y_augmented,
    batch_size=Augment_size,
    shuffle=False,
    save_to_dir='c:/Study25/_data/_save_img/02_mnist/',
).next()

exit()

x_trn = x_trn.reshape(-1,28,28,1)
x_tst = x_tst.reshape(-1,28,28,1)

x_trn = np.concatenate([x_trn, x_augmented])
y_trn = np.concatenate([y_trn, y_augmented])

y_trn = pd.get_dummies(y_trn)
y_tst = pd.get_dummies(y_tst)

print(x_trn.shape)
print(y_trn.shape)
print(x_tst.shape)
print(y_tst.shape)

#2. 모델 구성
path = './_save/keras36_cnn5/'
model = load_model(path + 'k36_0610_1525_0007-0.0538.h5')
""" loss : 0.13519825041294098
acc  : 0.9672999978065491
acc  : 0.9673

[Augment]
loss : 0.05730321630835533
acc  : 0.9851999878883362
acc  : 0.9852
"""
# model = Sequential()
# model.add(Conv2D(64,(3,3), strides=1 ,input_shape=(28,28,1), activation='relu'))  # 출력 형태 : 기존 형태 - kernel_size + 1
#                                                                # strides / 포복, kernel_size가 움직이는 픽셀의 크기
#                                                                # 통상적으로 strides는 kernel_size 보다는 작게 해야한다
# model.add(BatchNormalization())
# model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
# model.add(Dropout(0.1))

# model.add(BatchNormalization())
# model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
# model.add(Dropout(0.1))

# model.add(Flatten())                                           # Dense layer와 마찬가지로 Conv2D에서도 activation, Dropout 모두 사용가능

# model.add(BatchNormalization())
# model.add(Dense(units=16, activation='relu'))                  # 마찬가지로 성능이 향상 될 수 있다
# model.add(Dropout(0.1))

# model.add(BatchNormalization())
# model.add(Dense(units=16, input_shape=(16,), activation='relu'))
# model.add(Dropout(0.1))

# model.add(Dense(units=10, activation='softmax'))

#3. 컴파일 훈련
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['acc']
              )

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ES = EarlyStopping(monitor = 'val_loss', mode = 'min',
#                    patience= 50, verbose=1,                         # ES에서의 verbose = early stopping 지점을 알 수 있다
#                    restore_best_weights=True,
    
# )

################################# mpc 세이브 파일명 만들기 #################################
### 월일시 넣기
# import datetime

# date = datetime.datetime.now()
# # print(date)             # 2025-06-02 13:01:01.062409 >> 이 값에서 원하는 부분만 잘라오기
# #                                                    # >> 문자열로 변경 필요 (string)
# # print(type(date))       # <class 'datetime.datetime'>
# date = date.strftime('%m%d_%H%M')              
# # % + y : 년
#     # m : 월
#     # d : 일
#     # H : 시
#     # M : 분
    # S : 초

# print(date)             # 250602_130556
# print(type(date))       # <class 'str'>

### 모델 정보 넣기 (epoch, weight)
### 파일 저장

# path = './_save/keras36_cnn5/'
# filename = '{epoch:04d}-{val_loss:.4f}.h5' # 04d : 네자리 수 /// .f4 : 소수넷째자리
# filepath = "".join([path,'k36_',date, '_', filename])

# MCP = ModelCheckpoint(monitor='val_loss',
#                       mode='auto',
#                       save_best_only=True,
#                       verbose = 1,
#                       filepath= filepath # 확장자의 경우 h5랑 같음
#                                          # patience 만큼 지나기전 최저 갱신 지점        
#                       )
# Start = time.time()
# hist = model.fit(x_trn, y_trn,
#                  epochs = 5000,
#                  batch_size = 50,        # CNN에서 batch_size : 그림을 몇 장씩 한번에 처리하냐
#                  verbose = 3,
#                  validation_split = 0.2,
#                  callbacks = [ES, MCP])                    # 두 개 이상을 불러올수도 있다
# End = time.time()

# T = End - Start

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
# print("시간 :", T)

"""
[ x = (x - 127.5) / 510. ] + Batch_N    ############################
    loss : 0.03612019494175911
    acc  : 0.9908999800682068
    acc  : 0.9909
    시간 : 629.7389621734619
    
[augment]
"""