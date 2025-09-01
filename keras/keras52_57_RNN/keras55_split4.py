##########################################################################
#0. 준비
##########################################################################
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, GRU, SimpleRNN

from sklearn.model_selection import train_test_split

import numpy as np
import datetime

##########################################################################
#1. 데이터
##########################################################################

a = np.array(range(1,101))

#####################################
### 시계열 데이터 Spliting
TS = 10

def split(DS, TS):
    xy_a = []
    for i in range(len(DS) - TS):
        xy = DS[i : (i + TS + 1)]
        xy_a.append(xy)
        A = np.array(xy_a)
    return A[:,:-1], A[:,-1]

x, y = split(a,TS)
x = x.reshape(90,5,2)
y = y.reshape(-1,1)

# print(x)
# print(y)
# print(x.shape) (90, 10)
# print(y.shape) (90,)

##########################################################################
#2. 모델 구성
##########################################################################
model = Sequential()

model.add(LSTM(100, input_shape = (5,2), activation='relu', return_sequences=True))

model.add(GRU(100, input_shape = (5,2), activation='relu'))

model.add(Dense(100, activation='relu'))

model.add(Dense(100, activation='relu'))

model.add(Dense(50, activation='relu'))

model.add(Dense(25, activation='relu'))

model.add(Dense(1))

#####################################
### saveNum
date = datetime.datetime.now()
date = date.strftime('%m%d')
saveNum = f'{date}_0'

##########################################################################
#3. 컴파일 훈련
##########################################################################
model.compile(loss = 'mse',
              optimizer = 'adam')

### 가중치 불러오기
path = 'c:/Study25/_save/keras55/'
# model.load_weights(path + '0618_1.h5')

model.fit(x, y,
          epochs = 1000)

model.save(path + f'keras55_4_{saveNum}.h5')

##########################################################################
#4. 평가 예측
##########################################################################
loss = model.evaluate(x, y)

#####################################
### x_prd Spliting
x_prd = np.array(range(96,106))

def split(DS, TS):
    x_a = []
    for i in range(len(DS) - TS + 1):
        x = DS[i : (i + TS)]
        x_a.append(x)
        A = np.array(x_a)
    return A[:,:]

x_prd = split(x_prd, 10)
x_prd = x_prd.T.reshape(1,5,2)
# print(x_prd)
# print(x_prd.shape) (5, 2)

print('lss :', loss)
print('Rst :', model.predict(x_prd))

"""
lss : 0.00011569009075174108
Rst : [[106.066475]]
"""