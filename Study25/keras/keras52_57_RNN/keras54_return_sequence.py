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

model.add(LSTM(100, input_shape = (3,1), activation='relu', return_sequences=True))
                                                          # 단일 값이 아닌 Sequences르 모두 출력
                                                          # 3차원 시계열 데이터를 받아서 3차원 시계열 데이터를 출력
                                                          # 다시 RNN을 적용 가능
                                                          # 성능은 돌려봐야 안다

model.add(GRU(100, input_shape = (3,1), activation='relu'))

model.add(Dense(100, activation='relu'))

model.add(Dense(100, activation='relu'))

model.add(Dense(50, activation='relu'))

model.add(Dense(25, activation='relu'))

model.add(Dense(1))

# model.summary()

# exit()

'''
save = 0618_0
M : L(100) D(100 100 50 25)
[[50],[60],[70]] :  [[80.29755]]

save = 0618_2
M : L(100) G(100) D(100 100 50 25)
[[50],[60],[70]] : [[80.099846]]
'''
#####################################
### saveNum

date = datetime.datetime.now()
date = date.strftime('%m%d')
saveNum = f'{date}_2'

##########################################################################
#3. 컴파일 훈련
##########################################################################
model.compile(loss = 'mse',
              optimizer = 'adam')

### 가중치 불러오기
path = 'c:/Study25/_save/keras53/'
# model.load_weights(path + '0618_1.h5')

model.fit(x, y,
          epochs = 1000)

model.save(path + f'{saveNum}.h5')

##########################################################################
#4. 평가 예측
##########################################################################
loss = model.evaluate(x, y)

x_prd = np.array([50,60,70]).reshape(1,3,1)

rst = model.predict(x_prd)

print('loss             :', loss)
print('[[50],[60],[70]] :', rst)