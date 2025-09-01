## 20250530 실습_2 [ACC = 0.925]
# keras24_softmax3_fetch_covtype.copy

from sklearn.datasets import fetch_covtype

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

from imblearn.over_sampling import RandomOverSampler, SMOTE

RS = 55

#1. 데이터
DS = fetch_covtype()

x = DS.data
y = DS.target

y = y-1
y = to_categorical(y)

ros = RandomOverSampler(random_state=RS)
x, y = ros.fit_resample(x, y)

# smt = SMOTE(random_state=RS)
# x, y = smt.fit_resample(x, y)

# print(x.shape)  (581012, 54)
# print(y.shape)  (581012,)
# print(np.unique(y, return_counts=True))
# array([0, 1, 2, 3, 4, 5, 6]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510]

x_trn, x_tst, y_trn, y_tst = train_test_split(x, y,
                                                train_size = 0.9,
                                                shuffle = True,
                                                random_state = RS,
                                                # stratify=y,
                                                )

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
loss :  0.24401479959487915
ACC  :  0.9058398008346558
F1   :  0.9052730732828412

[MS]
[MS]
loss :  41.34183883666992
ACC  :  0.11481965333223343
F1   :  0.07622137557581775


[AS]
loss :  56.68803024291992
ACC  :  0.11271689087152481
F1   :  0.05686444012900819
[SS]
loss :  56.68803024291992
ACC  :  0.11271689087152481
F1   :  0.05686444012900819

[RS]
loss :  56.25147247314453
ACC  :  0.11157222837209702
F1   :  0.047799912983957235
'''

# print(x.shape) (581012, 54)
# print(y.shape) (581012, 7)

#2. 모델구성
def layer_tunig(a,b,c,d):
    model = Sequential()
    model.add(Dense(a, input_dim=54 , activation= 'relu'))
    # model.add(Dropout(0.2))
    model.add(BatchNormalization())
    
    model.add(Dense(b, activation= 'relu'))
    # model.add(Dropout(0.2))
    model.add(BatchNormalization())
    
    model.add(Dense(c, activation= 'relu'))
    # model.add(Dropout(0.2))
    model.add(BatchNormalization())
    
    model.add(Dense(c, activation= 'relu'))
    # model.add(Dropout(0.2))
    model.add(BatchNormalization())
    
    model.add(Dense(d, activation= 'relu'))
    # model.add(Dropout(0.2))
    model.add(BatchNormalization())
    
    model.add(Dense(d, activation= 'relu'))   
    # model.add(Dropout(0.2))
    model.add(BatchNormalization())
    
    model.add(Dense(d, activation= 'relu'))   
    # model.add(Dropout(0.2))
    model.add(BatchNormalization())
    
    model.add(Dense(7, activation= 'softmax'))
    return model

M = layer_tunig(30,40,30,20)
E = 100000
B = 5000
P = 100
V = 0.1

ES = EarlyStopping(monitor = 'val_loss',
                   mode = 'min',
                   patience = P,
                   restore_best_weights = True
                   )

################################# mpc 세이브 파일명 만들기 #################################
### 월일시 넣기
# import datetime

# path_MCP = './_save/keras28_mcp/10_covtype/'
# M = load_model(path_MCP + 'keras28_0604_1522_0149-0.0500.h5')

'''
loss :  0.24401479959487915
ACC  :  0.9058398008346558
F1   :  0.9052730732828412

loss :  0.24401479959487915
ACC  :  0.9058398008346558
F1   :  0.9052730732828412
'''

# date = datetime.datetime.now()
# # print(date)            
# # print(type(date))       
# date = date.strftime('%m%d_%H%M')              

# # print(date)             
# # print(type(date))

# filename = '{epoch:04d}-{val_loss:.4f}.h5'
# filepath = "".join([path_MCP,'keras28_',date, '_', filename])

# MCP = ModelCheckpoint(monitor='val_loss',
#                       mode='auto',
#                       save_best_only=True,
#                       filepath= filepath # 확장자의 경우 h5랑 같음
#                                          # patience 만큼 지나기전 최저 갱신 지점        
#                       )

#3. 컴파일 훈련
M.compile(loss='categorical_crossentropy',
          optimizer='adam',
          metrics = ['acc'])

# S_T = time.time()
H = M.fit(x, y,
          epochs=E, batch_size=B,
          verbose=1,
          validation_split=V,
          callbacks = [ES])
# E_T = time.time()

# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.figure(figsize=(9, 6))
# plt.title('---')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.plot(H.history['loss'], color = 'red', label = 'loss')
# plt.plot(H.history['val_loss'], color = 'green', label = 'val_loss')
# plt.legend(loc = 'upper right')
# plt.grid()

#4. 평가 예측
L = M.evaluate(x_tst, y_tst)
R = M.predict(x_tst)
R = np.round(R)
ACC = accuracy_score(y_tst, R)
F1 = f1_score(y_tst, R, average='macro')

print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print('loss : ', L[0])
print('ACC  : ', L[1])
print('F1   : ', F1)
# print('time : ', E_T - S_T,'초')
# plt.show()