from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
import time

import warnings

warnings.filterwarnings('ignore')


(x_trn, y_trn), (x_tst, y_tst) = reuters.load_data(num_words=1000, # 자연어 데이터에서 가저올 단어 사전의 개수, 빈도수가 높은 거 부터 1000개
                                                   test_split=0.2,
                                                   maxlen=2400,     # 최대 단어 개수
                                                   ) 

# print(x_trn)
# print(type(x_trn))
# <class 'numpy.ndarray'>
# print(x_trn.shape)
# maxlen = 100 : (4777,)
# maxlen = 200 : (7076,)
# maxlen = 생략 : (8982,)
# print(np.unique(y_trn))
# maxlen = 100 : [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]
# maxlen = 200 : [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]
# maxlen = 생략 : [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]


#### 패딩 ####
from tensorflow.keras.preprocessing.sequence import pad_sequences

def padding(a):
    a = pad_sequences(
        a,
        padding = 'pre',
        maxlen = 2400,
        truncating = 'pre'
    )
    
    return a

x_trn = padding(x_trn)
x_tst = padding(x_tst)
y_trn = to_categorical(y_trn)
y_tst = to_categorical(y_tst)

print('뉴스기사의 최대 길이 :', max(len(i) for i in x_trn))
# 간략화된 for문을 통한 x_trn의 최대 길이 뽑아내기 : 2376
print('뉴스기사의 최소 길이 :', min(len(i) for i in x_trn))
# 간략화된 for문을 통한 x_trn의 최소 길이 뽑아내기 : 13
# print('뉴스기사의 평균 길이 :', sum(map(len, x_trn)/len(x_trn)))
# 간략화된 for문을 통한 x_trn의 최소 길이 뽑아내기 : 145.53

# print(x_trn.shape) (8982, 2400)
# print(x_tst.shape) (2246, 2400)
# print(y_trn.shape) (8982, 46)
# print(y_tst.shape) (2246, 46)

'''
padding : maxlen = 2400
[1.9886705875396729, 0.7573463916778564]
'''

##########################################################################
#2. 모델 구성
##########################################################################
model = Sequential()

model.add(Embedding(1000, 200, input_length=2400))

model.add(LSTM(100))

model.add(Dense(50))

model.add(Dense(46, activation='softmax'))

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d')
saveNum = f'{date}_0'

####################################
### callbacks
# EarlyStopping
path = './_save/kears66/'

ES = EarlyStopping(monitor = 'acc',
                    mode = 'max',
                    patience = 10,
                    restore_best_weights = True)

MCP = ModelCheckpoint(monitor = 'acc',
                      mode = 'max',
                      save_best_only=True,
                      filepath= path + 'MCP_' + saveNum +'.h5')

##########################################################################
#3. 컴파일 훈련
##########################################################################
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

model.load_weights(path + 'MCP_0627.h5')

model.fit(x_trn, y_trn,
          epochs=1, batch_size=100,
          verbose=1,
          callbacks = [ES, MCP])

model.save_weights(path + f'weights_{saveNum}.h5')

##########################################################################
#4. 컴파일 훈련
##########################################################################
loss = model.evaluate(x_tst, y_tst)
y_prd = model.predict(x_tst)

y_prd = np.argmax(y_prd, axis=1)
y_prd = np.eye(46)[y_prd]

ACC = accuracy_score(y_tst, y_prd)

print(loss[0])
print(loss[1])
print(y_prd)
print(ACC)