## 20250530 실습_1 [ACC = 1]
## 41-9.copy

from sklearn.datasets import load_wine

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder


import numpy as np
import pandas as pd
import time

#1. 데이터
DS = load_wine()

x = DS.data
y = DS.target

x_trn, x_tst, y_trn, y_tst = train_test_split(x, y,
                                              train_size = 0.7,
                                              shuffle = True,
                                              random_state = 4165,
                                            #   stratify=y
)

y_trn = to_categorical(y_trn)
y_tst = to_categorical(y_tst)

# trn, tst에 y의 라벨 값이 불균형하게 들어갈수도!!!!
# 특히 데이터가 치중된 경우 모델이 애매해짐 >> stratify = y : y를 전략적으로 각 데이터를 나눠라

# print(x, y)
'''print(pd.value_counts(y))
1    71
0    59
2    48
'''
# print(x.shape, y.shape) (178, 13) (178,)

from tensorflow.keras.layers import Conv2D, Flatten
### reshape

x_trn = np.array(x_trn).reshape(-1,13,1,1)
x_tst = np.array(x_tst).reshape(-1,13,1,1)
# model.add(Conv2D(10, 1, padding='same', input_shape = (10,3,2)))
# model.add(Conv2D(10, 1))
# model.add(Flatten())

#2. 모델구성
from tensorflow.python.keras.layers import Conv1D

model = Sequential()
model.add(Conv1D(10, 1, padding='same', input_shape = (13,1,1)))
model.add(Conv1D(10, 1))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(40, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))

E = 1000000
B = 1
P = 30
V = 0.2
'''loss
0.8734215497970581
DO
0.5326288938522339
CNN
0.253219336271286
LSTM
0.7279239296913147
Conv1D
0.4058389961719513
'''

# ES = EarlyStopping(monitor='val_loss',
#                    mode='min',
#                    patience=P,
#                    restore_best_weights=True)

################################# mpc 세이브 파일명 만들기 #################################
### 월일시 넣기
# import datetime

# path_MCP = './_save/keras28_mcp/09_wine/'

# date = datetime.datetime.now()
# # print(date)            
# # print(type(date))       
# date = date.strftime('%m%d_%H%M')              

# # print(date)             
# # print(type(date))

# filename = '{epoch:04d}-{val_loss:.4f}.h5'
# filepath = "".join([path_MCP,'keras28_',date, '_', filename])

# MCP = ModelCheckpoint(monitor='val_loss',
#                       mode='auto',
#                       save_best_only=True,
#                       filepath= filepath # 확장자의 경우 h5랑 같음
#                                          # patience 만큼 지나기전 최저 갱신 지점        
#                       )

#3. 컴파일 훈련
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])
import time
start = time.time()
model.fit(x_trn, y_trn,
          epochs=100, batch_size=32,
          verbose=3,
          validation_split=V,
        #   callbacks = [ES, MCP]
          )

end = time.time()

#4. 평가 예측
loss = model.evaluate(x_tst,y_tst)
print(loss)
# import tensorflow as tf
# gpus = tf.config.list_physical_devices('GPU')
# print(gpus)

# if gpus:
#     print('GPU 있다!!!')
# else:
#     print('GPU 없다...')

# time = end - start
# print("소요시간 :", time)

'''
GPU 있다!!!
소요시간 : 3.489621639251709

GPU 없다...
소요시간 : 2.194779396057129
'''

'''
M = 30 80 30
E = 1000000
B = 10
P = 15
V = 0.2
loss :  0.1691625416278839
ACC  :  0.9722222222222222
F1   :  0.9688979039891819
'''