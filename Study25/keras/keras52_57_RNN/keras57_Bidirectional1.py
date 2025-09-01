##########################################################################
#0. 준비
##########################################################################
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Bidirectional

##########################################################################
#1. 데이터
##########################################################################
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

x = x.reshape(x.shape[0], x.shape[1], 1)

##########################################################################
#2. 모델 구성
##########################################################################
model = Sequential()
model.add(Bidirectional(SimpleRNN(units= 10), input_shape=(3, 1)))
model.add(Dense(units= 7, activation='relu'))
model.add(Dense(units= 1))
# 그냥 Bidr로는 레이어 제작 불가, 특정 레이어를 Bidr로 감싸줘야한다 : Rapping
# input layer를 Bidr로 잡고, RNN layer를 감쌈 == Bidr에 input을 넣어줘야함
model.summary()

'''
[ LSTM ] : Total params: 565
    lstm (LSTM)                    (None, 10)  480
    dense (Dense)                  (None, 7)   77
    dense_1 (Dense)                (None, 1)   8
 [ Bidi ] : Total params: 1,115
    bidirectional (Bidirectional)  (None, 20)  960
    dense (Dense)                  (None, 7)   147
    dense_1 (Dense)                (None, 1)   8

[ GRU ]  : Total params: 475
    gru (GRU)                      (None, 10)  390
    dense (Dense)                  (None, 7)   77
    dense_1 (Dense)                (None, 1)   8
 [ Bidi ] : Total params: 935
    bidirectional (Bidirectional)  (None, 20)  780
    dense (Dense)                  (None, 7)   147
    dense_1 (Dense)                (None, 1)   8

[ SRNN ] : Total params: 205
    simple_rnn (SimpleRNN)         (None, 10)  120
    dense (Dense)                  (None, 7)   77
    dense_1 (Dense)                (None, 1)   8
 [ Bidi ] : Total params: 395
    bidirectional (Bidirectional)  (None, 20)  240
    dense (Dense)                  (None, 7)   147
    dense_1 (Dense)                (None, 1)   8
'''

