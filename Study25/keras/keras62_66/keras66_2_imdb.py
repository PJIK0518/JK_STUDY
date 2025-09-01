# [실습] : ACC 0.85

from tensorflow.keras.datasets import imdb
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
import numpy as np
import numpy as pd

#1. 데이터

(x_trn, y_trn), (x_tst, y_tst) = imdb.load_data(
    num_words=1000,
)

""" print(x_trn) """
""" print(y_trn) [1 0 0 ... 0 1 0] """
""" print(x_trn.shape) (25000,) 몇 개짜린지는 모르겠지만 25000개의 list """
""" print(y_trn.shape) (25000,) 25000개의 이진 분류 값 25000개 """
""" print(np.unique(y_trn, return_counts=True)) (array([0, 1], dtype=int64), array([12500, 12500], dtype=int64)) """

#### 패딩 ####
from tensorflow.keras.preprocessing.sequence import pad_sequences

def padding(a):
    a = pad_sequences(
        a,
        padding = 'pre',
        maxlen = 2494,
        truncating = 'pre'
    )
    
    return a

x_trn = padding(x_trn)
x_tst = padding(x_tst)
print(x_trn.shape)
exit()

""" print('뉴스기사의 최대 길이 :', max(len(i) for i in x_trn)) 2494 """
""" print('뉴스기사의 최소 길이 :', min(len(i) for i in x_trn)) 2494 """

##########################################################################
#2. 모델 구성
##########################################################################
model = Sequential()

model.add(Embedding(1000, 100, input_length=2494))

model.add(LSTM(50))

model.add(Dense(30, activation='sigmoid'))

model.add(Dense(1, activation='sigmoid'))

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d')
saveNum = f'imdb_{date}_1_0'

''' 기록
0.861840009689331
0.86184
'''

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
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])

# model.load_weights(path + 'MCP_0627.h5')

model.fit(x_trn, y_trn,
          epochs=10000, batch_size=500,
          verbose=1,
          callbacks = [ES, MCP])

model.save_weights(path + f'imdb_weights_{saveNum}.h5')

##########################################################################
#4. 컴파일 훈련
##########################################################################
loss = model.evaluate(x_tst, y_tst)
y_prd = model.predict(x_tst)

y_prd = np.round(y_prd)

ACC = accuracy_score(y_tst, y_prd)

print(loss[0])
print(loss[1])
print(ACC)