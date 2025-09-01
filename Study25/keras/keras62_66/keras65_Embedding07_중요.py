# 06_ohe_과제까지는 Embedding 적용 X
# OneHotEncoding의 경우 0이 너무 많아서 데이터 용량, 성능, 시간적으로 너무 낭비
### Embedding

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

import numpy as np
import time

import warnings

warnings.filterwarnings('ignore')

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
x = padding_x
y = labels
##########################################################################
#2. 모델 구성
##########################################################################
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM

model = Sequential()
""" EBD.1 == 기본
model.add(Embedding(input_dim=31, output_dim=100, input_length=5))
# input_dim : 인식하려는 word_index, 단어 사전, 말뭉치의 개수 = 단어의 N
# output_dim : 글자를 변환하려는 Vector list의 길이 = units (조절 가능)
# input_length : 한 문장을 구성하는 최대 길이 = columns, maxlen
# params(EBD) : 단어 하나당 벡터 하나로 전환하는 연산 > 단어수 * 벡터 길이 > ID * OD """

''' EBD.2 == input_length를 명시하지 않아도 모델 구성 및 훈련은 가능, BUT. 틀리면 ERROR
model.add(Embedding(input_dim=31, output_dim=100))
'''

''' EBD.3 == input_dim 사이즈가 크면 가능하지만 작아지면 특정 단어를 사용하지 않고 진행 > 성능 감소
          (버전이 높아지면 훈련 불가)
model.add(Embedding(input_dim=110, output_dim=100))
'''

''' EBD.4 == 파라미터 명 생략시 첫번째 두번째까지는 input_D, output_D까지는 순서 확정 input_l은 ㄴㄴㄴ
             input_length는 알면쓰고 모르면 그냥 비워두고 1로 해도 가능은 하지만 손해
model.add(Embedding(31,100)) # 가능
model.add(Embedding(31,100,5)) # ValueError: Could not interpret initializer identifier: 5
model.add(Embedding(31,100, input_length=1)) # Warning뜨면서 가능 (column수가 달라도 1만 가능)
model.add(Embedding(31,100, input_length=5)) # 가능
model.add(Embedding(31,100, input_length=2)) # ValueError: Input 0 of layer "sequential" is incompatible with the layer: expected shape=(None, 2), found shape=(1, 5)
model.add(Embedding(31,100, input_length=6)) # ValueError: Input 0 of layer "sequential" is incompatible with the layer: expected shape=(None, 6), found shape=(1, 5)
'''

model.add(LSTM(16))
model.add(Dense(1))

model.summary()

""" EBD.1
 Layer (type)                Output Shape              Param #
=================================================================
 embedding (Embedding)       (None, 5, 100)            3100
 lstm (LSTM)                 (None, 16)                7488
 dense (Dense)               (None, 1)                 17
=================================================================
Total params: 10,605
Trainable params: 10,605
Non-trainable params: 0
"""

""" EBD.2
 Layer (type)                Output Shape              Param #
=================================================================
 embedding (Embedding)       (None, None, 100)         3100
 lstm (LSTM)                 (None, 16)                7488
 dense (Dense)               (None, 1)                 17
=================================================================
Total params: 10,605
Trainable params: 10,605
Non-trainable params: 0
"""
# exit()
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

####################################
### callbacks
## EarlyStopping
# ES = EarlyStopping(monitor = 'acc',
#                     mode = 'max',
#                     patience = P,
#                     restore_best_weights = True)


##########################################################################
#3. 컴파일 훈련
##########################################################################
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])

model.fit(x, y,
          epochs=100, batch_size=1,
          verbose=2)


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

loss = model.evaluate(x, y)
rslt = model.predict(x_prd)
rslt = np.round(rslt)

print(loss) # [0.0, 1.0]
print(rslt) # [[1.]]