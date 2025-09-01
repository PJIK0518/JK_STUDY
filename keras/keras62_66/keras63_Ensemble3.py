# 63_2.copy
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

y1 = np.array(range(2001, 2101))
# (100,)           화성의 화씨 온도

y2 = np.array(range(13001, 13101))

# 이전 방식 : x1, x2를 합쳐서 모델 제작 후 > y 예측

x1_trn, x1_tst, x2_trn, x2_tst, x3_trn, x3_tst, y1_trn, y1_tst, y2_trn, y2_tst = train_test_split(x1_datasets, x2_datasets, x3_datasets, y1, y2,
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
### X_1
Input1 = Input(shape=(2,))

Dense1 = Dense(30, activation='relu', name='ibm1')(Input1)
Dense2 = Dense(30, activation='relu', name='ibm2')(Dense1)
Dense3 = Dense(30, activation='relu', name='ibm3')(Dense2)
Dense4 = Dense(20, activation='relu', name='ibm4')(Dense3)
Outpt1 = Dense(10, activation='relu', name='ibm5')(Dense4) 

#####################################
### X_2
Input2 = Input(shape=(3,))

Dens21 = Dense(30, activation='relu', name='ibm21')(Input2)
Dens22 = Dense(20, activation='relu', name='ibm22')(Dens21)
Outpt2 = Dense(10, activation='relu', name='ibm23')(Dens22)

#####################################
### X_3
Input3 = Input(shape=(4,))

Dens31 = Dense(30, activation='relu', name='ibm31')(Input3)
Dens32 = Dense(20, activation='relu', name='ibm32')(Dens31)
Outpt3 = Dense(10, activation='relu', name='ibm33')(Dens32)

#####################################
### Merge : 모델1, 모델2의 OutPut을 이어 받는다
#            여러가지 방법이 있지만 가장 흔하게 쓰는건 Concateante

from keras.layers.merge import concatenate, Concatenate
Merge1 = Concatenate(axis=1)([Outpt1, Outpt2, Outpt3])
Merge2 = Dense(30, activation='relu', name = 'mg1')(Merge1)
Merge3 = Dense(20, activation='relu', name = 'mg2')(Merge2)
OtptLM = Dense(1, name = 'merge')(Merge3)

#####################################
### Y_1 : 모델1, 모델2의 OutPut을 이어 받는다
#            여러가지 방법이 있지만 가장 흔하게 쓰는건 Concateante

rslt12 = Dense(30, activation='relu', name = 'rt11')(OtptLM)
rslt13 = Dense(20, activation='relu', name = 'rt12')(rslt12)
OtptL1 = Dense(1, name = 'last1')(rslt13)

#####################################
### Y_2 : 모델1, 모델2의 OutPut을 이어 받는다
#            여러가지 방법이 있지만 가장 흔하게 쓰는건 Concateante

rslt21 = Dense(10, activation='relu', name = 'rt21')(OtptLM)
rslt22 = Dense(30, activation='relu', name = 'rt22')(rslt21)
rslt23 = Dense(10, activation='relu', name = 'rt23')(rslt22)
OtptL2 = Dense(1, name = 'last2')(rslt23)

model = Model(inputs = [Input1, Input2, Input3], outputs = [OtptL1, OtptL2])

E, B, P, V = (10000, 10, 50, 0.2)
# Ensemble에 들어가는 데이터1, 2의 열은 상관이 없지만, 행은 같아야 Concat가능

### >> 
#####################################
### callbacks
ES = EarlyStopping(monitor='val_loss',
                   mode = 'min',
                   patience = P,
                   restore_best_weights= True)

##########################################################################
#3. 컴파일 훈련
##########################################################################
model.compile(loss = ['mse'],
              optimizer = 'adam',
              metrics=['mae']) # mean_av

model.fit([x1_trn, x2_trn, x3_trn], [y1_trn, y2_trn],
          epochs = E,
          verbose = 3,
          validation_split= 0.2,
          callbacks=[ES])
##########################################################################
#4. 평가 예측
##########################################################################
loss = model.evaluate([x1_tst, x2_tst, x3_tst], [y1_tst, y2_tst])

x1_prd = np.array([range(100,106), range(400,406)]).T
x2_prd = np.array([range(200,206), range(510,516), range(249,255)]).T
x3_prd = np.array([range(100,106), range(400,406), range(177,183), range(133,139)]).T

result = model.predict([x1_prd, x2_prd, x3_prd])

print(loss)
print(result)
print(np.round(result))
""" print(result)
[array([[2096.2603],
       [2098.781 ],
       [2101.3015],
       [2103.8218],
       [2106.3425],
       [2108.8633]], dtype=float32), array([[13069.875],
       [13080.597],
       [13091.32 ],
       [13102.041],
       [13112.763],
       [13123.485]], dtype=float32)]
loss: 6.9319 - last1_loss: 0.0289 - last2_loss: 6.9030
               last1_mae: 0.0352 - last2_mae: 0.4809"""
               
# 모델 전체 loss / 첫번째 output에 대한 loss / 두번째 output에 대한 loss