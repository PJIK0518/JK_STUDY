## 20250530 실습_1 [ACC = 1]
## keras28_MCP_save_09_wine.copy

from sklearn.datasets import load_wine

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder


import numpy as np
import pandas as pd
import time

#1. 데이터
DS = load_wine()

x = DS.data
y = DS.target

x_trn, x_tst, y_trn, y_tst = train_test_split(x, y,
                                              train_size = 0.7,
                                              shuffle = True,
                                              random_state = 4165,
                                            #   stratify=y
)

y_trn = to_categorical(y_trn)
y_tst = to_categorical(y_tst)

# trn, tst에 y의 라벨 값이 불균형하게 들어갈수도!!!!
# 특히 데이터가 치중된 경우 모델이 애매해짐 >> stratify = y : y를 전략적으로 각 데이터를 나눠라

# print(x, y)
'''print(pd.value_counts(y))
1    71
0    59
2    48
'''
# print(x.shape, y.shape) (178, 13) (178,)

### scaling
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
# MS = MinMaxScaler()
# MS.fit(x_trn)
# x_trn = MS.transform(x_trn)
# x_tst = MS.transform(x_tst)

# AS = MaxAbsScaler()
# AS.fit(x_trn)
# x_trn = AS.transform(x_trn)
# x_tst = AS.transform(x_tst)

# SS = StandardScaler()
# SS.fit(x_trn)
# x_trn = SS.transform(x_trn)
# x_tst = SS.transform(x_tst)

RS = RobustScaler()
RS.fit(x_trn)
x_trn = RS.transform(x_trn)
x_tst = RS.transform(x_tst)

'''
loss :  0.2084583044052124
ACC  :  0.8888888888888888
F1   :  0.9052042160737813

[MS]
loss :  0.03350666165351868
ACC  :  0.9629629629629629
F1   :  0.9703557312252965

[AS]
loss :  0.04637931287288666
ACC  :  0.9814814814814815
F1   :  0.9841179807146908

[SS]
loss :  0.005429644137620926
ACC  :  1.0
F1   :  1.0

[RS]
loss :  0.23581159114837646
ACC  :  0.9629629629629629
F1   :  0.9622256499244836

'''

#2. 모델구성
model = Sequential()
model.add(Dense(40, input_dim = 13, activation ='relu'))
model.add(Dense(60, activation = 'relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))

E = 1000000
B = 1
P = 30
V = 0.2

ES = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience=P,
                   restore_best_weights=True)

################################# mpc 세이브 파일명 만들기 #################################
### 월일시 넣기
import datetime

path_MCP = './_save/keras28_mcp/09_wine/'

date = datetime.datetime.now()
# print(date)            
# print(type(date))       
date = date.strftime('%m%d_%H%M')              

# print(date)             
# print(type(date))

filename = '{epoch:04d}-{val_loss:.4f}.h5'
filepath = "".join([path_MCP,'keras28_',date, '_', filename])

MCP = ModelCheckpoint(monitor='val_loss',
                      mode='auto',
                      save_best_only=True,
                      filepath= filepath # 확장자의 경우 h5랑 같음
                                         # patience 만큼 지나기전 최저 갱신 지점        
                      )

#3. 컴파일 훈련
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

ST = time.time()

model.fit(x_trn, y_trn,
          epochs=E, batch_size=B,
          verbose=3,
          validation_split=V,
          callbacks = [ES])

ET = time.time()

#4. 평가 예측
L = model.evaluate(x_tst, y_tst)
R = model.predict(x_tst)
R = np.round(R)
F1 = f1_score(y_tst, R, average = 'macro')
ACC = accuracy_score(y_tst,R)

print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print("loss : ", L[0])
print("ACC  : ", ACC)
print("F1   : ", F1)

'''
M = 30 80 30
E = 1000000
B = 10
P = 15
V = 0.2
loss :  0.1691625416278839
ACC  :  0.9722222222222222
F1   :  0.9688979039891819
'''