# 07_2_1.cpoy

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
import numpy as np

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9],
              [9,8,7,6,5,4,3,2,1,0]])
y = np.array([1,2,3,4,5,6,7,8,9,10])
# x = np.array([1,1,9],[2,1.1,8]...)
x = np.transpose(x)

print(x.shape) # (3, 10) > (10, 3)
print(y.shape) # (10,)

#2-1. 모델구성 : 순차형 모델, Sequential
model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(9))
model.add(Dropout(0.3))
model.add(Dense(8))
model.add(Dropout(0.2))
model.add(Dense(7))
model.add(Dense(1))

epochs = 100

""" model.summary()
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 10)                40
 dropout (Dropout)           (None, 10)                0
 dense_1 (Dense)             (None, 9)                 99
 dense_2 (Dense)             (None, 8)                 80
 dropout_1 (Dropout)         (None, 8)                 0
 dense_3 (Dense)             (None, 7)                 63
 dense_4 (Dense)             (None, 1)                 8
=================================================================
Total params: 290
Trainable params: 290
Non-trainable params: 0
"""
# Dropout에 의해 모델 전체 연산량은 영향 X, 훈련시 Batch 단위로만 실질적으로 감소


#2-2. 모델 구성 : 함수형 모델, Model

# 함수형 모델은 모델 input_layer 자체를 먼저 구성해야한다.
# 모든 구성을 마친 후 model의 inputs(시점)와 outputs(종점)을 정의
# 지금은 순차형 모델을 그대로 구현했지만, 점프 역행 등의 구성을 추가 가능
input_1 = Input(shape=(3,))
dense_1 = Dense(10, name='ys1')(input_1)        # 연산하거나 모델 구성할 때, 이름 정해놓을 수 있다.
dense_2 = Dense(9, name='ys1')(dense_1)
dropO_1 = Dropout(0.3)(dense_2)
dense_3 = Dense(8)(dropO_1)
dropO_2 = Dropout(0.2)(dense_3)
dense_4 = Dense(7)(dropO_2)
OutPut1 = Dense(1)(dense_4)
model_2 = Model(inputs=input_1, outputs=OutPut1)

""" model_2.summary()
 Layer (type)                Output Shape              Param #
=================================================================
 input_1 (InputLayer)        [(None, 3)]               0
 dense_5 (Dense)             (None, 10)                40
 dense_6 (Dense)             (None, 9)                 99
 dropout_2 (Dropout)         (None, 9)                 0
 dense_7 (Dense)             (None, 8)                 80
 dropout_3 (Dropout)         (None, 8)                 0
 dense_8 (Dense)             (None, 7)                 63
 dense_9 (Dense)             (None, 1)                 8
=================================================================
Total params: 290
Trainable params: 290
Non-trainable params: 0 """