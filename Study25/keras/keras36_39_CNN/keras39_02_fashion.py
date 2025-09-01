# 36-5.copy

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.datasets import fashion_mnist

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np
import time
##########################################################################
#1. 데이터
##########################################################################
(x_trn, y_trn), (x_tst, y_tst) = fashion_mnist.load_data()
# print(x_trn.shape) # (60000, 28, 28)
# print(x_tst.shape) # (10000, 28, 28)
# print(y_trn.shape) # (60000,)
# print(y_tst.shape) # (10000,)

#####################################
### x reshape 
x_trn = x_trn.reshape(60000,28,28,1)
x_tst = x_tst.reshape(x_tst.shape[0], x_tst.shape[1], x_tst.shape[2], 1)
                                            # 이렇게 입력해도 똑같이 인식함
# print(x_trn.shape, x_tst.shape) # (60000, 28, 28, 1) (10000, 28, 28, 1)

#####################################
### y OneHot
y_trn = pd.get_dummies(y_trn)
y_tst = pd.get_dummies(y_tst)
# print(y_trn.shape, y_tst.shape) (60000, 10) (10000, 10)

#####################################
### Scaling
x_trn = (x_trn-127.5)/(510.0)
x_tst = (x_tst-127.5)/(510.0)

##########################################################################
#2. 모델 구성
### 모델 불러오기
path = './_save/keras39/fashion/'
model = load_model(path + 'k39_0612_1000.h5')
#####################################

# model = Sequential()
# model.add(Conv2D(100, 2, input_shape=(28,28,1), activation='relu',
#                  strides=1, padding='valid'))
# model.add(Dropout(0.2))

# model.add(BatchNormalization())
# model.add(Conv2D(filters=80, kernel_size=2, activation='relu',
#                  strides=1, padding='valid'))
# model.add(Dropout(0.2))

# model.add(BatchNormalization())
# model.add(Conv2D(filters=60, kernel_size=2, activation='relu',
#                  strides=1, padding='valid'))
# model.add(Dropout(0.2))

# model.add(BatchNormalization())
# model.add(Conv2D(filters=40, kernel_size=2, activation='relu',
#                  strides=1, padding='valid'))
# model.add(Dropout(0.2))

# model.add(BatchNormalization())
# model.add(Conv2D(filters=20, kernel_size=2, activation='relu',
#                  strides=1, padding='valid'))
# model.add(MaxPool2D(4))
# model.add(Dropout(0.2))

# model.add(Flatten())                                           

# model.add(BatchNormalization())
# model.add(Dense(units=100, activation='relu'))                  
# model.add(Dropout(0.2))

# model.add(BatchNormalization())
# model.add(Dense(units=80, activation='relu'))                  
# model.add(Dropout(0.2))

# model.add(BatchNormalization())
# model.add(Dense(units=50, activation='relu'))                  
# model.add(Dropout(0.2))

# model.add(BatchNormalization())
# model.add(Dense(units=20, input_shape=(16,), activation='relu'))
# model.add(Dropout(0.2))

# model.add(Dense(units=10, activation='softmax'))

##########################################################################
#3. 컴파일 훈련
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['acc']
              )


# ES = EarlyStopping(monitor = 'val_acc', mode = 'max',
#                    patience= 50, verbose=1,                         # ES에서의 verbose = early stopping 지점을 알 수 있다
#                    restore_best_weights=True,
    
# )
################################# mpc 세이브 파일명 만들기 #################################
### 월일시 넣기
# import datetime

# date = datetime.datetime.now()
# date = date.strftime('%m%d_%H%M')              

### 모델 정보 넣기 (epoch, weight)
### 파일 저장

# filename = '{epoch:04d}-{val_loss:.4f}.h5' # 04d : 네자리 수 /// .f4 : 소수넷째자리
# filepath = "".join([path,'k39_',date, '.h5'])

# MCP = ModelCheckpoint(monitor='val_acc',
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

'''
loss : 0.22302168607711792
acc  : 0.9302999973297119
acc  : 0.9303
시간 : 1882.5303266048431

loss : 0.20780806243419647
acc  : 0.9316999912261963
acc  : 0.9317
시간 : 1904.9725253582


'''


