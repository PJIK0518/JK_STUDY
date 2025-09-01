# 52-1.copy

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU

import numpy as np

# ##########################################################################
# #1. 데이터
# ##########################################################################

datasets = np.array([1,2,3,4,5,6,7,8,9,10])     
# 데이터 의미를 부여하기 나름 시계열 데이터로 해석 할 수 있다.
# 실제로 시계열 데이터는 2차원으로 제공될 수도 있음
# 또한, y값도 제공되지 않고, 우리가 직접 부여해야한다.
# 이 때, 우리는 Time Step에 따라서 잘라 줘야한다.            
      
x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9]])          
# datasets을 Time Step = 3 으로 자른 경우
# 통상적으로 Time Step이 크면 모델 성능이 증가하지만 최적의 값은 몇인지 알아야한다
y = np.array([4,5,6,7,8,9,10])

""" 데이터 해석
x[0] > y[0],
x[1] > y[1],
x[2] > y[2],
x[3] > y[3],
x[4] > y[4]가 나올것이다
"""
# print(x.shape)  (7, 3)    : (N-TS, TS) >> 3차원 데이터가 아니다? feature가 빠짐
# print(y.shape)  (7,)      : (N-TS, )

x = x.reshape(x.shape[0], x.shape[1], 1)
# print(x.shape)  (7, 3, 1) : (N-TS, TS, F)
#                             (batch_size, time_step, feature)
""" print(x)
x = np.array([[[1],[2],[3]],
              [[2],[3],[4]],
              [[3],[4],[5]],
              [[4],[5],[6]],
              [[5],[6],[7]],
              [[6],[7],[8]],
              [[7],[8],[9]]])
"""
# time_step 만큼 잘라진 데이터는 모델 훈련을 할 때, feature 단위로 훈련 진행

##########################################################################
# #2. 모델 구성
##########################################################################
model = Sequential()
# model.add(SimpleRNN(units=10, input_shape = (3,1), activation='relu'))
# model.add(LSTM(units=10, input_shape = (3,1), activation='relu'))
model.add(GRU(units=10, input_shape = (3,1), activation='relu'))
model.add(Dense(5, activation='relu'))      
model.add(Dense(1))

model.summary()
##########################################################################
# Param(simpleRNN) = feature*unit + unit*unit + bias*unit
                #  = 1차 연산 + 순환 연산 + 바이어스
                #  = (featuer + unit + bias)*unit
# simplerRNN의 경우 Time Step이 커지면 초기 연산에 대한 결과가 소실될 위험
''' [ summary.simpleRNN ] feature*unit + unit*unit + bias*unit
 Layer (type)                Output Shape              Param #   
=================================================================
 simple_rnn (SimpleRNN)      (None, 10)                120       
 dense (Dense)               (None, 5)                 55        
 dense_1 (Dense)             (None, 1)                 6
=================================================================
Total params: 181
Trainable params: 181
Non-trainable params: 0
'''
##########################################################################


##########################################################################
# Param(LSTM) = (feature*unit + unit*unit + bias*unit)*4
# LSTM Architecture
# : Input gate(sigm) + Forget gate(sigm + tanh) + Output Gate(sigm) + Cell State > 연산량 네 배
# : 초기 값 > IG + FG + OG > cell state, hidden state 형성 > 다음 Time Step
''' [ summary.LSTM ]
 Layer (type)                Output Shape              Param #
=================================================================
 lstm (LSTM)                 (None, 10)                480
 dense (Dense)               (None, 5)                 55
 dense_1 (Dense)             (None, 1)                 6
=================================================================
Total params: 541
Trainable params: 541
Non-trainable params: 0
'''
##########################################################################


##########################################################################
# Param(GRU) = (feature*unit + unit*unit + 2*bias*unit)*3
# Pythorch에서는 bias를 하나로 잡고 계산하지만, python 같은 경우 2개로 계산
''' [ summary.GRU ]
 Layer (type)                Output Shape              Param #   
=================================================================
 gru (GRU)                   (None, 10)                390       
 dense (Dense)               (None, 5)                 55        
 dense_1 (Dense)             (None, 1)                 6
=================================================================
Total params: 451
Trainable params: 451
Non-trainable params: 0
'''