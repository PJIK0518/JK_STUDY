from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization

import numpy as np

#2. 모델
model = Sequential()
model.add(Dense(3, input_dim = 1))
model.add(Dense(2))
model.add(Dense(4))
# model.add(BatchNormalization())
model.add(Dense(1))

model.summary()         # 모델의 종 연산량을 계산해줌
                        # > 실제 이론적인 량이랑은 다름
'''
 Layer (type)                Output Shape              Param #
=================================================================
 dense (Dense)               (None, 3)                 6

 dense_1 (Dense)             (None, 2)                 8

 dense_2 (Dense)             (None, 4)                 12

 dense_3 (Dense)             (None, 1)                 5

=================================================================
Total params: 31
Trainable params: 31
Non-trainable params: 0
'''

## 모델이 너무 느리거나, 사전학습 모델을 땡겨 쓸 때 답답하면 좀 Tunning할 때 활용