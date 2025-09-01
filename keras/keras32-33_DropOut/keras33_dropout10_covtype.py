## 20250530 실습_2 [ACC = 0.925]
# 31-10
# keras24_softmax3_fetch_covtype.copy

from sklearn.datasets import fetch_covtype

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

from sklearn.datasets import get_data_home
# print(get_data_home())

# exit()

# from imblearn.over_sampling import RandomOverSampler, SMOTE

RS = 55

#1. 데이터
DS = fetch_covtype()

x = DS.data
y = DS.target

y = y-1
y = to_categorical(y)

# ros = RandomOverSampler(random_state=RS)
# x, y = ros.fit_resample(x, y)

# smt = SMOTE(random_state=RS)
# x, y = smt.fit_resample(x, y)

# print(x.shape)  (581012, 54)
# print(y.shape)  (581012,)
# print(np.unique(y, return_counts=True))
# array([0, 1, 2, 3, 4, 5, 6]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510]

x_trn, x_tst, y_trn, y_tst = train_test_split(x, y,
                                                train_size = 0.9,
                                                shuffle = True,
                                                random_state = RS,
                                                # stratify=y,
                                                )

# print(x.shape) (581012, 54)
# print(y.shape) (581012, 7)

#2. 모델구성
def layer_tunig(a,b,c,d):
    model = Sequential()
    model.add(Dense(a, input_dim=54 , activation= 'relu'))
    # model.add(Dropout(0.2))
    model.add(BatchNormalization())
    
    model.add(Dense(b, activation= 'relu'))
    # model.add(Dropout(0.2))
    model.add(BatchNormalization())
    
    model.add(Dense(c, activation= 'relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    
    model.add(Dense(c, activation= 'relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    
    model.add(Dense(d, activation= 'relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    
    model.add(Dense(d, activation= 'relu'))   
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    
    model.add(Dense(d, activation= 'relu'))   
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    
    model.add(Dense(7, activation= 'softmax'))
    return model

M = layer_tunig(30,40,30,20)
E = 100000
B = 5000
P = 100
V = 0.1

'''loss
0.5709723830223083
0.7167189717292786
'''
# ES = EarlyStopping(monitor = 'val_loss',
#                    mode = 'min',
#                    patience = P,
#                    restore_best_weights = True
#                    )

################################# mpc 세이브 파일명 만들기 #################################
### 월일시 넣기
# import datetime

# path_MCP = './_save/keras28_mcp/10_covtype/'

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
M.compile(loss='categorical_crossentropy',
          optimizer='adam',
          metrics = ['acc'])

# S_T = time.time()

import time
start = time.time()
H = M.fit(x, y,
          epochs=10, batch_size=32,
          verbose=1,
          validation_split=V,
        #   callbacks = [ES, MCP]
          )
end = time.time()
# E_T = time.time()

# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.figure(figsize=(9, 6))
# plt.title('---')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.plot(H.history['loss'], color = 'red', label = 'loss')
# plt.plot(H.history['val_loss'], color = 'green', label = 'val_loss')
# plt.legend(loc = 'upper right')
# plt.grid()

#4. 평가 예측
loss = M.evaluate(x_tst,y_tst)
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
소요시간 : 2301.598026752472

GPU 없다...
소요시간 : 234.9175410270691
'''
# print('time : ', E_T - S_T,'초')
# plt.show()