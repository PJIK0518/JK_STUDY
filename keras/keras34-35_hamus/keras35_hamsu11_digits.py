# 33_11.copy

##########################################################################
#0. 준비
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, OneHotEncoder, StandardScaler
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
import datetime
import numpy as np
import pandas as pb
import time

RS = 42
##########################################################################
#1 데이터
DS = load_digits()

x = DS.data
y = DS.target

# print(x.shape)  (1797, 64)
# print(y.shape)  (1797,)
""" print(np.unique(y, return_counts=True))
(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))
"""
#####################################
## OneHot
One_y = OneHotEncoder()
y = y.reshape(-1,1)
y = One_y.fit_transform(y).toarray()
""" print(np.unique(y, return_counts=True))
(array([<1797x10 sparse matrix of type '<class 'numpy.float64'>'
        with 1797 stored elements in Compressed Sparse Row format>],
      dtype=object), array([1], dtype=int64))
"""

x_trn, x_tst, y_trn, y_tst = train_test_split(x, y,
                                              train_size=0.8,
                                              shuffle= True,
                                              random_state=RS,
                                              stratify=y)

# print(x.shape) (1797, 64)
# print(y.shape) (1797, 10)

#####################################
## Scaler
## MinMaxScaler
# MS = MinMaxScaler()
# MS_col=['aaa','bbb','ccc']
# MS.fit(AAA[MS_col])
# AAA[MS_col] = MS.transfrom(AAA[MS_col])

## StandardScaler
# SS = StandardScaler()
# SS_col=['aaa','bbb','ccc']
# SS.fit(AAA[SS_col])
# AAA[SS_col] = MS.transfrom(AAA[SS_col])

## MaxAbsScaler
# MbS = MaxAbsScaler()
# MbS_col=['aaa','bbb','ccc']
# MbS.fit(AAA[SS_col])
# AAA[MbS_col] = MbS.transfrom(AAA[MbS_col])

## RobustScaler
# RbS = RobustScaler()
# RbS_col=['aaa','bbb','ccc']
# RbS.fit(AAA[RbS_col])
# AAA[RbS_col] = RbS.transfrom(AAA[RbS_col])

##########################################################################
#2 모델구성
#####################################
## 불러오기
# path = './_save/keras30/'
# M = load_model(path + '')
#####################################
# def f(a,b,c):
#         model = Sequential()
#         model.add(Dense(a, input_dim=64 , activation='relu'))
#         # model.add(Dropout())
#         model.add(Dense(b, activation='relu'))
#         # model.add(Dropout())
#         model.add(Dense(c, activation='relu'))
#         # model.add(Dropout())
#         model.add(Dense(10, activation='softmax'))
#         return model
# M = f(64,64,64)

input_1 = Input(shape=(64,))

dense_1 = Dense(64, activation ='relu')(input_1)
# dropO_1 = Dropout()(dense_1)

dense_2 = Dense(64, activation ='relu')(dense_1)
# dropO_2 = Dropout()(dense_2)

dense_3 = Dense(64, activation ='relu')(dense_2)
# dropO_3 = Dropout()(dense_3)

OutPut1 = Dense(10, activation ='softmax')(dense_3)

M = Model(inputs=input_1, outputs=OutPut1)

E, B, P, V = (100000,32,10,0.2)

''' loss
'''

#####################################
## 저장 정보
date = datetime.datetime.now()
date = date.strftime('%m%d')

saveNum = f'{date}_2'
fit_info = '{epochs:04d}_{val_acc:.4f}'

#####################################
## Callbacks
ES = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=P,
    restore_best_weights=True
)

path = './_save/keras30/digit/MCP/'

MCP = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    filepath="".join([path, 'MCP_', saveNum, '.h5'])
)

##########################################################################
#3 컴파일, 훈련
M.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

S = time.time()
H = M.fit(x_trn, y_trn,
          epochs=E, batch_size=B,
          verbose=2,
          validation_split=V,
          callbacks = [ES, MCP])
E = time.time()
T = E - S

##########################################################################
#4. 평가 예측
L = M.evaluate(x_tst, y_tst)
R = M.predict(x_tst)
R = np.round(R)
F1 = f1_score(y_tst,R, average='macro')

#####################################
## 그래프
# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.figure(figsize=(9, 6))
# plt.title('Digit')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.plot(H.history['loss'], color = 'red', label = 'loss')
# plt.plot(H.history['val_loss'], color = 'green', label = 'val_loss')
# plt.legend(loc = 'upper right')
# plt.grid()

print('loss :', L[0])
print('ACC  :', L[1])
print('F1sc :', F1)
print('time :', T)
# plt.show()