## 20250530 실습_2 [ACC = 0.925]
from sklearn.datasets import fetch_covtype

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
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

''' Tuning // 최고 기록 [ ACC  :  0.9290659427642822 ]
batchnormalization, randomoversampler
RS = 55
 M = 30,40,30,30,20,20,20
 E = 100000
 B = 5000
 P = 100
 V = 0.1
    stratify=y / restore_best_weights = True
    loss :  0.2628900706768036
    ACC  :  0.8993449807167053
    F1   :  0.8993354436683927
    time :  647.1641089916229 초

    # stratify=y / restore_best_weights = False
    loss :  0.1837620586156845
    ACC  :  0.9290659427642822 *****
    F1   :  0.9288012787875235
    time :  495.603303194046 초

    stratify=y / restore_best_weights = False
    loss :  0.24738697707653046
    ACC  :  0.9067625999450684
    F1   :  0.9061364038827034
    time :  505.8845248222351 초

    # stratify=y / restore_best_weights = True
    loss :  0.19093120098114014
    ACC  :  0.9280725717544556
    F1   :  0.9279353611405469
    time :  650.3706712722778 초
'''

#3. 컴파일 훈련
M.compile(loss='categorical_crossentropy',
          optimizer='adam',
          metrics = ['acc'])

S_T = time.time()
H = M.fit(x, y,
          epochs=E, batch_size=B,
          verbose=1,
          validation_split=V,
          callbacks = [ES])
E_T = time.time()

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(9, 6))
plt.title('---')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.plot(H.history['loss'], color = 'red', label = 'loss')
plt.plot(H.history['val_loss'], color = 'green', label = 'val_loss')
plt.legend(loc = 'upper right')
plt.grid()

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
print('time : ', E_T - S_T,'초')
plt.show()