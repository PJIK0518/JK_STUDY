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

y = np.array(range(2001, 2101))
# (100,)           화성의 화씨 온도

# 이전 방식 : x1, x2를 합쳐서 모델 제작 후 > y 예측

x1_trn, x1_tst, x2_trn, x2_tst, y_trn, y_tst = train_test_split(x1_datasets, x2_datasets, y,
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
# Ensemble에서는 최종 Node의 갯수가 y 값이랑 같을 필요가 없다
# : 모델1, 2의 최종 layer는 모델 3의 input으로 연결되는 Hidden이 되기 때문

# Model1 = Model(inputs = Input1, outputs = Outpt1)
# 최종 모델은 1, 2의 데이터를 받아서 사용하기 때문에 모델 정의는 필요없다
""" Model1.summary() input = (2,) output = (50,)
 Layer (type)                Output Shape              Param #
=================================================================
 input_1 (InputLayer)        [(None, 2)]               0
 ibm1 (Dense)                (None, 10)                30
 ibm2 (Dense)                (None, 20)                220
 ibm3 (Dense)                (None, 30)                630
 ibm4 (Dense)                (None, 40)                1240
 ibm5 (Dense)                (None, 50)                2050
=================================================================
Total params: 4,170
Trainable params: 4,170
Non-trainable params: 0
"""

#####################################
### 모델 2
Input2 = Input(shape=(3,))

Dens21 = Dense(30, name='ibm21')(Input2)
Dens22 = Dense(20, name='ibm22')(Dens21)
Outpt2 = Dense(10, name='ibm23')(Dens22)

# Model2 = Model(inputs = Input2, outputs = Outpt2)
# 최종 모델은 1, 2의 데이터를 받아서 사용하기 때문에 모델 정의는 필요없다
""" Model2.summary() input = (3,) output = (30,)
 Layer (type)                Output Shape              Param #
=================================================================
 input_2 (InputLayer)        [(None, 3)]               0
 ibm21 (Dense)               (None, 100)               400
 ibm22 (Dense)               (None, 50)                5050
 ibm23 (Dense)               (None, 30)                1530
=================================================================
Total params: 6,980
Trainable params: 6,980
Non-trainable params: 0
"""

#####################################
### 모델 3 : 모델1, 모델2의 OutPut을 이어 받는다
#            여러가지 방법이 있지만 가장 흔하게 쓰는건 Concateante

from keras.layers.merge import concatenate, Concatenate

Merge1 = concatenate([Outpt1, Outpt2], name = 'mg1')

Merge2 = Dense(100, activation='relu', name = 'mg2')(Merge1)
Merge3 = Dense(50, activation='relu', name = 'mg3')(Merge2)
OutptL = Dense(1, activation='relu', name = 'last')(Merge3)

model = Model(inputs = [Input1, Input2], outputs = OutptL)

E, B, P, V = (1000000, 1, 500, 0.2)
""" Model.summary()
 Layer (type)         Output Shape  Param #  Connected to
==============================================================================
 input_1 (InputLayer) [(None, 2)]   0        []
 ibm1 (Dense)         (None, 10)    30       ['input_1[0][0]']
 ibm2 (Dense)         (None, 20)    220      ['ibm1[0][0]']
 input_2 (InputLayer) [(None, 3)]   0        []
 ibm3 (Dense)         (None, 30)    630      ['ibm2[0][0]']
 ibm21 (Dense)        (None, 100)   400      ['input_2[0][0]']
 ibm4 (Dense)         (None, 40)    1240     ['ibm3[0][0]']
 ibm22 (Dense)        (None, 50)    5050     ['ibm21[0][0]']
 ibm5 (Dense)         (None, 50)    2050     ['ibm4[0][0]']
 ibm23 (Dense)        (None, 30)    1530     ['ibm22[0][0]']
 mg1 (Concatenate)    (None, 80)    0        ['ibm5[0][0]', 'ibm23[0][0]']
 mg2 (Dense)          (None, 40)    3240     ['mg1[0][0]']
 mg3 (Dense)          (None, 20)    820      ['mg2[0][0]']
 last (Dense)         (None, 1)     21       ['mg3[0][0]']
==================================================================================================
Total params: 15,231
Trainable params: 15,231
Non-trainable params: 0 """
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

model.fit([x1_trn, x2_trn], y_trn,
          epochs = E,
          verbose = 3,
          validation_split= 0.2,
          callbacks=[ES])
##########################################################################
#4. 평가 예측
##########################################################################
loss = model.evaluate([x1_tst, x2_tst], y_tst)

x1_prd = np.array([range(100,106), range(400,406)]).T
x2_prd = np.array([range(200,206), range(510,516), range(249,255)]).T

result = model.predict([x1_prd, x2_prd])

print(result)