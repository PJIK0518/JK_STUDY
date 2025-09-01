# 46_02.copy

from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, load_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import time

RS = 42
##########################################################################
#1. 데이터
##########################################################################
path_NP = 'C:/Study25/_data/tensor_cert/rps/'

x = np.load(path_NP + 'x_trn.npy')
y = np.load(path_NP + 'y_trn.npy')
y = np.argmax(y, axis=1)
# print(x.shape)
# print(y.shape)
# print(np.unique(y))

x_trn,x_tst,y_trn,y_tst = train_test_split(x, y,
                                           train_size=0.7,
                                           shuffle=True,
                                           random_state=RS)

##########################################################################
#2. 모델 구성
##########################################################################
### 모델 불러오기
path_MCP = './_data/tensor_cert/rps/MCP/'
path_S = 'C:/Study25/_data/tensor_cert/rps/save/'
# model = load_model(path_MCP + 'MCP_0613_0_3.h5')

model = Sequential()
model.add(Conv2D(100, 5, input_shape=(150, 150, 3),
                 padding='valid', activation='relu'))
model.add(MaxPool2D(2))
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(Conv2D(50, 4, padding='valid'))
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(Conv2D(10, 3, padding='valid'))
model.add(MaxPool2D(2))

model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(BatchNormalization())
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(Dense(20, activation='relu'))

model.add(Dense(3, activation='softmax'))

E, B, P, V = (10000, 32, 50, 0.2)

'''
save : 0616_0_0
loss : 0.08671111613512039
ACC  : 0.9772357940673828

save : 0616_0_1
loss : 0.004666629247367382
ACC  : 0.9983739852905273
'''

#####################################
### 저장 설정

D = datetime.datetime.now()
D = D.strftime('%m%d')
saveNum = f'{D}_0_1'

#####################################
### callbacks

ES = EarlyStopping(
    monitor='val_acc',
    mode='max',
    patience=P,
    restore_best_weights=True
)

MCP = ModelCheckpoint(
    monitor='val_acc',
    mode='max',
    filepath="".join([path_MCP,'MCP_',saveNum,'.h5']),
    save_best_only=True
)

##########################################################################
#3. 컴파일 훈련
##########################################################################
model.compile(
    loss = 'sparse_categorical_crossentropy',
    optimizer = 'adam',
    metrics=['acc']
)

#####################################
### 가중치 불러오기
path_W = 'C:/Study25/_data/tensor_cert/rps/weights/'
# model.load_weights(path_W + f'weight_0616_0_0.h5')

H = model.fit(
    x_trn, y_trn,
    epochs = E,
    batch_size = B,
    verbose = 1,
    validation_split=0.2,
    callbacks=[ES, MCP]
)

#####################################
### 모델 및 가중치 저장
model.save(path_S + f'save_{saveNum}.h5')
model.save_weights(path_W + f'weights_{saveNum}.h5')

##########################################################################
#4. 평가 예측
##########################################################################
LSAC = model.evaluate(x_tst, y_tst)

y_pred = model.predict(x_tst)
y_pred = np.argmax(y_pred, axis=1)  

ACC = accuracy_score(y_tst,y_pred)

#####################################
### 그래프
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(9, 6))
plt.title('human_horse')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.plot(H.history['loss'], color = 'red', label = 'loss')
plt.plot(H.history['val_loss'], color = 'green', label = 'val_loss')
plt.legend(loc = 'upper right')
plt.grid()

print('save :', saveNum)
print('loss :', LSAC[0])
print('ACC  :', LSAC[1])
plt.show()