#53.copy

##########################################################################
#0. 준비
##########################################################################
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM, Conv1D, Dropout, BatchNormalization

from sklearn.model_selection import train_test_split

import numpy as np
import datetime

##########################################################################
#1. 데이터
##########################################################################
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
              [5,6,7], [6,7,8], [7,8,9], [8,9,10],
              [9,10,11], [10,11,12],
              [20,30,40], [30,40,50], [40,50,60]
              ])

y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x = x.reshape(x.shape[0],x.shape[1],1)
# print(x.shape) (13, 3, 1)

##########################################################################
#2. 모델 구성
##########################################################################
model = Sequential()

model.add(Conv1D(100,2, input_shape = (3,1), activation='relu'))
model.add(Flatten())

model.add(Dense(100, activation='relu'))

model.add(Dense(100, activation='relu'))

model.add(Dense(50, activation='relu'))

model.add(Dense(25, activation='relu'))

model.add(Dense(1))


'''
save = 0618_0
M : 100 100 100 50 25
[[50],[60],[70]] :  [[80.29755]]

save = 0623_0
[[50],[60],[70]] :  [[79.964294]]
'''
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

model.fit(x, y,
          epochs = 1000)

path = 'c:/Study25/_save/keras53/'
model.save(path + f'{saveNum}.h5')

##########################################################################
#4. 평가 예측
##########################################################################
loss = model.evaluate(x, y)

x_prd = np.array([50,60,70])

x_prd = x_prd.reshape(1,3,1)

rst = model.predict(x_prd)

print('[[50],[60],[70]] : ', rst)