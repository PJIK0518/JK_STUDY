# 63_1.copy
##########################################################################
#0. 준비
##########################################################################
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import numpy as np
import time

##########################################################################
#1. 데이터
##########################################################################
x1_datasets = np.array([range(100), range(301,401)]).T
# (100, 2)              삼성전자 종가, 하이닉스 종가

x2_datasets = np.array([range(101,201), range(411,511), range(150,250)]).transpose()
# (100, 3)              원유, 환율, 금 시세

x3_datasets = np.array([range(100), range(301,401), range(77,177), range(33,133)]).T

y = np.array(range(2001, 2101))
# (100,)           화성의 화씨 온도

# 이전 방식 : x1, x2를 합쳐서 모델 제작 후 > y 예측

x1_trn, x1_tst, x2_trn, x2_tst, x3_trn, x3_tst, y_trn, y_tst = train_test_split(x1_datasets, x2_datasets, x3_datasets, y,
                                                                train_size=0.7,
                                                                random_state=42)
# train_test_split의 경우 데이터 안에 순서대로 넣고 정의하면 됨
# a1, a2, b1, b2, c1, c2 = train_test_split(A, B, C)
# >> A를 a1 a2, B를 b1 b2, C를 c1 c2로 분할
# print(x1_trn.shape) (70, 2)
# print(x1_tst.shape) (30, 2)
# print(x2_trn.shape) (70, 3)
# print(x2_tst.shape) (30, 3)
# print(y_trn.shape)  (70,)
# print(y_tst.shape)  (30,)

##########################################################################
#2. 모델 구성
##########################################################################
### 모델 1
Input1 = Input(shape=(2,))

Dense1 = Dense(50, activation='relu', name='ibm1')(Input1)
Dense2 = Dense(40, activation='relu', name='ibm2')(Dense1)
Dense3 = Dense(30, activation='relu', name='ibm3')(Dense2)
Dense4 = Dense(20, activation='relu', name='ibm4')(Dense3)
Outpt1 = Dense(10, activation='relu', name='ibm5')(Dense4) 

#####################################
### 모델 2
Input2 = Input(shape=(3,))

Dens21 = Dense(30, activation='relu', name='ibm21')(Input2)
Dens22 = Dense(20, activation='relu', name='ibm22')(Dens21)
Outpt2 = Dense(10, activation='relu', name='ibm23')(Dens22)

#####################################
### 모델 3
Input3 = Input(shape=(4,))

Dens31 = Dense(30, activation='relu', name='ibm31')(Input3)
Dens32 = Dense(20, activation='relu', name='ibm32')(Dens31)
Outpt3 = Dense(10, activation='relu', name='ibm33')(Dens32)

#####################################
### 모델 M : 모델1, 모델2의 OutPut을 이어 받는다
#            여러가지 방법이 있지만 가장 흔하게 쓰는건 Concateante

from keras.layers.merge import concatenate, Concatenate

Merge1 = Concatenate(axis=1)([Outpt1, Outpt2, Outpt3])

Merge2 = Dense(30, activation='relu', name = 'mg2')(Merge1)
Merge3 = Dense(20, activation='relu', name = 'mg3')(Merge2)
OutptL = Dense(1, activation='relu', name = 'last')(Merge3)

model = Model(inputs = [Input1, Input2, Input3], outputs = OutptL)

E, B, P, V = (10000, 10, 50, 0.2)
# Ensemble에 들어가는 데이터1, 2의 열은 상관이 없지만, 행은 같아야 Concat가능
#####################################
### callbacks
ES = EarlyStopping(monitor='val_loss',
                   mode = 'min',
                   patience = P,
                   restore_best_weights= True)

##########################################################################
#3. 컴파일 훈련
##########################################################################
model.compile(loss = 'mse',
              optimizer = 'adam')

model.fit([x1_trn, x2_trn, x3_trn], y_trn,
          epochs = E,
          verbose = 3,
          validation_split= 0.2,
          callbacks=[ES])
##########################################################################
#4. 평가 예측
##########################################################################
loss = model.evaluate([x1_tst, x2_tst, x3_tst], y_tst)

x1_prd = np.array([range(100,106), range(400,406)]).T
x2_prd = np.array([range(200,206), range(510,516), range(249,255)]).T
x3_prd = np.array([range(100,106), range(400,406), range(177,183), range(133,139)]).T

result = model.predict([x1_prd, x2_prd, x3_prd])

print(result)