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
#####################################

model = Sequential()
model.add(Conv2D(100, 2, input_shape=(28,28,1), activation='relu',
                 strides=1, padding='valid'))
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(Conv2D(filters=80, kernel_size=2, activation='relu',
                 strides=1, padding='valid'))
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(Conv2D(filters=60, kernel_size=2, activation='relu',
                 strides=1, padding='valid'))
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(Conv2D(filters=40, kernel_size=2, activation='relu',
                 strides=1, padding='valid'))
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(Conv2D(filters=20, kernel_size=2, activation='relu',
                 strides=1, padding='valid'))
model.add(MaxPool2D(4))
model.add(Dropout(0.2))

model.add(Flatten())                                           

model.add(BatchNormalization())
model.add(Dense(units=100, activation='relu'))                  
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(Dense(units=80, activation='relu'))                  
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(Dense(units=50, activation='relu'))                  
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(Dense(units=20, input_shape=(16,), activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(units=10, activation='softmax'))
#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau

best_op = []
best_lr = 0
best_sc = -10000


model.compile(loss = 'mse', optimizer = Adam(learning_rate=0.01))


ES = EarlyStopping(monitor='val_loss',
                mode= 'min',
                patience= 50,
                restore_best_weights= True)

RLR = ReduceLROnPlateau(monitor = 'val_loss',
                        mode = 'auto',
                        patience = 10,
                        verbose = 1,
                        factor = 0.5)
                    # patience 만큼 갱신되지 않으면 해당 비율만큼 lr 하강(곱하기)


hist = model.fit(x_trn, y_trn, epochs = 10000, batch_size = 32,
        verbose=2,
        validation_split=0.2,
        callbacks = [ES, RLR])

loss = model.evaluate(x_tst, y_tst)
results = model.predict([x_tst])

R2 = accuracy_score(y_tst, results)

print('scr :',R2)
print('lss :',loss)
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


