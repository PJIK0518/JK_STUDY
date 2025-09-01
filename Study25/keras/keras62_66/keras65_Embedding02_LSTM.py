# 65_01.copy

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

import numpy as np
import time
docs = [
    '너무 재미있다',
    '참 최고에요',
    '참 잘만든 영화에요',
    '추천하고 싶은 영화입니다',
    '한 번 더 보고 싶어요',
    '글쎄',
    '별로에요',
    '생각보다 지루해요',
    '연기가 어색해요',
    '재미없어요',
    '너무 잼미없다',
    '참 재밋네요',
    '석준이 바보',
    '준희 잘생겼다',
    '이삭이 또 구라친다',
]

labels = np.array([1,
                   1,
                   1,
                   1,
                   1,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0,
                   1,
                   0,
                   1,
                   0])

# 1 : 증정 / 0 : 부정

token = Tokenizer()

token.fit_on_texts(docs)

""" print(token.word_index)
{'참': 1, '너무': 2, '재미있다': 3, '최고에요': 4, '잘만든': 5,
 '영화에요': 6, '추천하고': 7, '싶은': 8, '영화입니다': 9,
 '한': 10, '번': 11, '더': 12, '보고': 13, '싶어요': 14,
 '글쎄': 15, '별로에요': 16, '생각보다': 17, '지루해요': 18,
 '연기가': 19, '어색해요': 20, '재미없어요': 21, '잼미없다': 22,
 '재밋네요': 23, '석준이': 24, '바보': 25, '준희': 26,
 '잘생겼다': 27, '이삭이': 28, '또': 29, '구라친다': 30} """

x = token.texts_to_sequences(docs)

# print(x) : list 안에 list가 들어잇는 상태
#            꺼내서 써야하는데..
#            list 마다 길이가 다름
#            최대 길이로 맞추기 : padding with 0 >> 모델이나 데이터의 형태를 보고 어디 넣을지 결정
#         or 최소 길이로 맞추기? : 작은 데이터에서는 소실량이 치명적
# [[2, 3], [1, 4], [1, 5, 6], [7, 8, 9],
#  [10, 11, 12, 13, 14], [15], [16], [17, 18],
#  [19, 20], [21], [2, 22], [1, 23], [24, 25],
#  [26, 27], [28, 29, 30]]

#### 패딩 ####
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 0으로 각 list의 길이를 맞춰주기
padding_x =pad_sequences(
           x,
           padding = 'pre',   # 'post' : 0을 어디 넣을 것인가 앞 vs 뒤 / Default : pre
           maxlen = 5,        #  최대 길이
           truncating = 'pre' # 'post' : 최대길이를 초과하면 어디부터 짜를것인가
)

""" print(padding_x) padding = pre / maxlen = 5 / truncating = pre
[[ 0  0  0  2  3]
 [ 0  0  0  1  4]
 [ 0  0  1  5  6]
 [ 0  0  7  8  9]
 [10 11 12 13 14]
 [ 0  0  0  0 15]
 [ 0  0  0  0 16]
 [ 0  0  0 17 18]
 [ 0  0  0 19 20]
 [ 0  0  0  0 21]
 [ 0  0  0  2 22]
 [ 0  0  0  1 23]
 [ 0  0  0 24 25]
 [ 0  0  0 26 27]
 [ 0  0 28 29 30]] """
""" print(padding_x) padding = XXX / maxlen = 5 / truncating = pre
[[ 0  0  0  2  3]
 [ 0  0  0  1  4]
 [ 0  0  1  5  6]
 [ 0  0  7  8  9]
 [10 11 12 13 14]
 [ 0  0  0  0 15]
 [ 0  0  0  0 16]
 [ 0  0  0 17 18]
 [ 0  0  0 19 20]
 [ 0  0  0  0 21]
 [ 0  0  0  2 22]
 [ 0  0  0  1 23]
 [ 0  0  0 24 25]
 [ 0  0  0 26 27]
 [ 0  0 28 29 30]] """
""" print(padding_x) padding = pre / maxlen = 7 / truncating = pre
[[ 0  0  0  0  0  2  3]
 [ 0  0  0  0  0  1  4]
 [ 0  0  0  0  1  5  6]
 [ 0  0  0  0  7  8  9]
 [ 0  0 10 11 12 13 14]
 [ 0  0  0  0  0  0 15]
 [ 0  0  0  0  0  0 16]
 [ 0  0  0  0  0 17 18]
 [ 0  0  0  0  0 19 20]
 [ 0  0  0  0  0  0 21]
 [ 0  0  0  0  0  2 22]
 [ 0  0  0  0  0  1 23]
 [ 0  0  0  0  0 24 25]
 [ 0  0  0  0  0 26 27]
 [ 0  0  0  0 28 29 30]] """
""" print(padding_x) padding = pre / maxlen = 3 / truncating = pre
[[ 0  2  3]
 [ 0  1  4]
 [ 1  5  6]
 [ 7  8  9]
 [12 13 14]
 [ 0  0 15]
 [ 0  0 16]
 [ 0 17 18]
 [ 0 19 20]
 [ 0  0 21]
 [ 0  2 22]
 [ 0  1 23]
 [ 0 24 25]
 [ 0 26 27]
 [28 29 30]] """

x = np.array(padding_x)
y = labels

x = x.reshape(-1,5,1)

x_trn, x_tst, y_trn, y_tst = train_test_split(x, y,
                                              train_size=0.8,
                                              shuffle =True,
                                              random_state=32)
# print(x.shape) (15, 5)
# print(y.shape) (15,)


#### [실습] : 이삭이 참 잘생겼다 
##########################################################################
#2. 모델 구성
##########################################################################

from tensorflow.keras.layers import Dropout, BatchNormalization

def LT(a,b,DO):
    model = Sequential()
    model.add(LSTM(a, input_shape=(5,1) , activation='relu'))
    model.add(Dropout(DO))

    model.add(BatchNormalization())
    model.add(Dense(a, activation='relu'))
    model.add(Dropout(DO))
    
    model.add(BatchNormalization())
    model.add(Dense(a, activation='relu'))
    model.add(Dropout(DO))

    model.add(BatchNormalization())
    model.add(Dense(b, activation='sigmoid'))
    
    model.add(Dense(1, activation='sigmoid'))
    return model

M = LT(70,35,0.2)
E, B, P, V = (100000,5,50,0.2)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

####################################
### callbacks
## EarlyStopping
ES = EarlyStopping(monitor = 'acc',
                    mode = 'max',
                    patience = P,
                    restore_best_weights = True)

##########################################################################
#3. 컴파일 훈련
##########################################################################
M.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])
M.fit(x_trn, y_trn,
          epochs=E, batch_size=B,
          verbose=3,
          callbacks = [ES])


##########################################################################
#4. 컴파일 훈련
##########################################################################
prd = ['이삭이 참 잘생겼다']

x_prd = token.texts_to_sequences(prd)
x_prd = pad_sequences(
           x_prd,
           padding = 'pre',   
           maxlen = 5,  
           truncating = 'pre' 
)

x_prd = x_prd.reshape(-1,5,1)


loss = M.evaluate(x_tst, y_tst)
rslt = M.predict(x_prd)
rslt = np.round(rslt)

print(rslt)