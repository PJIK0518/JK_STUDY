### 41-1.copy

import sklearn as sk
# print(sk.__version__)   1.1.3

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
import numpy as np
import time

#1. 데이터
DS = load_boston()

x = DS.data
y = DS.target

### print(x.shape, y.shape) (506, 13) (506,)

x_trn, x_tst, y_trn, y_tst = train_test_split(x, y,
                                              train_size=0.8,
                                              shuffle=True,
                                              random_state=333)

### scaling
from sklearn.preprocessing import MinMaxScaler
MS = MinMaxScaler()
MS.fit(x_trn)
x_trn = MS.transform(x_trn)
x_tst = MS.transform(x_tst)

from tensorflow.keras.layers import Conv2D, Flatten
### reshape

x_trn = x_trn.reshape(-1,13,1,1)
x_tst = x_tst.reshape(-1,13,1,1)
# model.add(Conv2D(10, 1, padding='same', input_shape = (13,1,1)))
# model.add(Conv2D(10, 1))
# model.add(Flatten())

#2. 모델
from tensorflow.python.keras.layers import Conv1D

model = Sequential()
model.add(Conv1D(10, 1, padding='same', input_shape = (13,1,1)))
# model.add(Dropout(0.1))
model.add(Conv1D(10, 1))
# model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(12))
# model.add(Dropout(0.1))
model.add(Dense(13))
model.add(Dropout(0.1))
model.add(Dense(1))

''' loss
23.134387969970703
[DO]
23.04183578491211
[CNN]
22.289579391479492
[LSTM]
35.09996032714844
[Conv1D]
22.802509307861328
'''
#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')

# ES = EarlyStopping(monitor= 'val_loss',               # 어떤 수치를 모니터링 할 것인가
#                    mode= 'min',                       # 최대값 : max, 알아서 찾아라 : auto
#                    patience= 50,                      # 몇 번이나 참을것이냐, 통상적으로 커야지만 너무 크면 시간 낭비가 될 수도
#                    restore_best_weights= True)       # 가장 최소 지점을 저장 한다 // Default : False, False가 성적이 더 좋을수도...

################################# mpc 세이브 파일명 만들기 #################################
### 월일시 넣기
import datetime

path_MCP = './_save/keras28_mcp/01_boston/'

date = datetime.datetime.now()
# print(date)            
# print(type(date))       
date = date.strftime('%m%d_%H%M')              

# print(date)             
# print(type(date))

# filename = '{epoch:04d}-{val_loss:.4f}.h5'
# filepath = "".join([path_MCP,'keras28_',date, '_', filename])

# MCP = ModelCheckpoint(monitor='val_loss',
#                       mode='auto',
#                       save_best_only=True,
#                       filepath= filepath # 확장자의 경우 h5랑 같음
#                                          # patience 만큼 지나기전 최저 갱신 지점        
#                       )

start = time.time()
hist = model.fit(x_trn, y_trn,
                 epochs = 200,
                 batch_size = 32,
                 verbose = 2,
                 validation_split = 0.2,
                #  callbacks = [ES, MCP]
                 )                    # 두 개 이상을 불러올수도 있다
end = time.time()
# path = './_save/keras27_mcp/'
# model.save(path + 'keras27_3_save.h5')
# model.save_weights(path + 'keras26_5_save2.h5')

#4. 평가,예측
loss = model.evaluate(x_tst, y_tst)
results = model.predict(x_tst)
r2 = r2_score(y_tst, results)
rmse = np.sqrt(loss)

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
소요시간 : 8.325046300888062

GPU 있다!!!
소요시간 : 10.001005172729492

GPU 없다...
소요시간 : 4.489137887954712
'''