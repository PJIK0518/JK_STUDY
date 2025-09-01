# 36_4.copy

# 사진보고 정리
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.datasets import mnist

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np
import time

from tensorflow.keras.layers import GlobalAvgPool2D

#2. 모델 구성
model = Sequential()
model.add(Conv2D(64,(3,3), strides=2 ,input_shape=(28,28,1)))  # 출력 형태 : 기존 형태 - kernel_size + 1
                                                               # strides / 포복, kernel_size가 움직이는 픽셀의 크기
model.add(Conv2D(filters=64, kernel_size=(3,3)))
model.add(Conv2D(filters=32, kernel_size=(3,3)))
model.add(GlobalAvgPool2D())
# model.add(Flatten())
# model.add(Dense(units=16))
# model.add(Dense(units=16))
model.add(Dense(units=10, activation='softmax'))
""" model.summary() : GAP : 필터별로 평균때려서 사용
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 13, 13, 64)        640       
                                                                 
 conv2d_1 (Conv2D)           (None, 11, 11, 64)        36928     
                                                                 
 conv2d_2 (Conv2D)           (None, 9, 9, 32)          18464     
                                                                 
 global_average_pooling2d (  (None, 32)                0          > 실질적으로 (None, 1, 1, 32)
                                                                  > tensorflow에서는 알아서 1,1을 처리
                                                                  > torch에서는 직접해야한다...
 GlobalAveragePooling2D)                                         
                                                                 
 dense (Dense)               (None, 10)                330       
                                                                 
=================================================================
Total params: 56362 (220.16 KB)
Trainable params: 56362 (220.16 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________ """

""" model.summary() strides = 1 (Default)
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 26, 26, 64)        640
 conv2d_1 (Conv2D)           (None, 24, 24, 32)        18464
 conv2d_2 (Conv2D)           (None, 22, 22, 16)        4624
 flatten (Flatten)           (None, 7744)              0
 dense (Dense)               (None, 16)                123920
 dense_1 (Dense)             (None, 16)                272
 dense_2 (Dense)             (None, 10)                170
=================================================================
Total params: 148,090
Trainable params: 148,090
Non-trainable params: 0 """
""" model.summary() strides = 2
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 13, 13, 64)        640       
 conv2d_1 (Conv2D)           (None, 11, 11, 32)        18464     
 conv2d_2 (Conv2D)           (None, 9, 9, 16)          4624
 flatten (Flatten)           (None, 1296)              0
 dense (Dense)               (None, 16)                20752
 dense_1 (Dense)             (None, 16)                272
 dense_2 (Dense)             (None, 10)                170
=================================================================
Total params: 44,922
Trainable params: 44,922
Non-trainable params: 0 """

