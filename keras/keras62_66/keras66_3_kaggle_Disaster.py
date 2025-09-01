# https://www.kaggle.com/competitions/nlp-getting-started/data

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Embedding, Dropout, BatchNormalization, LSTM
from tensorflow.keras.models import Sequential

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import re

RS = 807
# ==========================================================================
# 데이터
# ==========================================================================
# 경로 
path = './_data/kaggle/disaster/'
path_MCP = './_data/kaggle/disaster/MCP/'
path_sub = './_data/kaggle/disaster/sub/'

trn_csv = pd.read_csv(path + 'train.csv', index_col = 0)
prd_csv = pd.read_csv(path + 'test.csv', index_col = 0)
sub_csv = pd.read_csv(path + 'sample_submission.csv')

# 데이터 전처리
# 1,2,3 columns을 concatenate
#       : How : Columns[1]_name + Columns[1]_내용 + Columns[2]_name + Columns[2]_내용 + Columns[3]_name + Columns[3]_내용

def merge_text(df):
    df['keyword'] = df['keyword'].fillna('unknown')
    df['location'] = df['location'].fillna('unknown').str.lower()
    df['text'] = df['text'] + 'keyword' + df['keyword'] + 'location' + df['location']
    # df['text'] = df['text'].apply(lambda x: re.sub(r'http\S+|www\S+|https\S+', '', x, flags=re.MULTILINE))
    df['text'] = df['text']
    return df

trn_csv = merge_text(trn_csv)
prd_csv = merge_text(prd_csv)

# print(trn_csv.info())
# print(prd_csv.info())

Token = Tokenizer()
Token.fit_on_texts(trn_csv['text'])

x = Token.texts_to_sequences(trn_csv['text'])
# x = np.array(x)
y = trn_csv['target']

y = np.array(y)
y = y.reshape(-1,1)

x_prd = Token.texts_to_sequences(prd_csv['text'])
# x_prd = np.array(x_prd)

# print(max(len(i) for i in x)) 
# print(min(len(i) for i in x))
# print(max(len(i) for i in x_prd)) 
# print(min(len(i) for i in x_prd))

# print(x.shape) (7613,)
# print(pd.value_counts(y))
# 0    4342
# 1    3271
# print(x_prd.shape) (3263,)

def padding(data):
    data = pad_sequences(data,
                         padding = 'pre',
                         maxlen= 40,
                         truncating= 'pre')
    return data

x = padding(x)
x_prd = padding(x_prd)

x_trn, x_tst, y_trn, y_tst = train_test_split(x, y,
                                              train_size=0.9,
                                              shuffle=True,
                                              random_state=RS,
                                              stratify = y)

vocab_size = len(Token.word_index) + 1 

# ==========================================================================
# 모델
# ==========================================================================

def Layer_Tuning(a, b, c, DO):
    model = Sequential()
    model.add(Embedding(vocab_size, a, input_length=40))
    
    model.add(LSTM(b, activation = 'relu'))
    model.add(Dropout(DO))

    model.add(BatchNormalization())
    model.add(Dense(c, activation = 'sigmoid'))
    
    model.add(Dense(1, activation = 'sigmoid'))
    
    return model

M = Layer_Tuning(50,30,10,0.2)

Epochs, Batch, Patience \
    = (100000, 10000, 10)

# =====================================
# 파일명 저장
date = datetime.datetime.now()
date = date.strftime('%m%d')

saveNum = f'{date}_1_3'

# =====================================
# callbacks
ES = EarlyStopping(monitor = 'acc',
                   mode = 'max',
                   patience= Patience,
                   restore_best_weights=True)

MCP = ModelCheckpoint(monitor='acc',
                      mode = 'max',
                      filepath = path_MCP + f'MCP_{saveNum}.h5')

# ==========================================================================
# 컴파일 훈련
# ==========================================================================
M.compile(loss = 'binary_crossentropy',
          optimizer = 'adam',
          metrics = ["acc"])

# M.load_weights(path_MCP + 'MCP_0630_1_0.h5')

H = M.fit(x_trn, y_trn,
      epochs = Epochs,
      verbose = 1,
      callbacks = [ES, MCP])


# ==========================================================================
# 평가 훈련
# ==========================================================================
loss = M.evaluate(x_tst, y_tst)

# =====================================
# 그래프
# plt.figure(figsiez = (9, 6))
# plt.title('LOSS')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.plot()

print(loss)

# =====================================
# 파일 송출

y_prd = M.predict(x_prd)
y_prd = np.round(y_prd).astype(int).flatten()

print(np.unique(y_prd, return_counts=True))

sub_csv['target'] = y_prd
sub_csv.to_csv(path_sub + f'sample_sub_{saveNum}.csv', index = False)