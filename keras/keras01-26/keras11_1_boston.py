# sklearn에서 제공하는 boston의 옛날 집 값 데이터

import sklearn as sk
# print(sk.__version__)   # 1.6.1 -> 1.1.3 (datasets 찾으러)

from sklearn.datasets import load_boston

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np

#1. 데이터
dataset = load_boston()
# print(dataset)         
    # 데이터를 불러 쓸때, 프린트해서 유의사항 확인
    # data : 통상적으로 x
    # target : 통상적으로 y
    # feature : column 수
    # Number of instance : 데이터의 n수
# print(dataset.DESCR)
    # DESCR, describe? 묘사하다 : dataset의 특징정리
# print(dataset.feature_names)
    # ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']
    
x = dataset.data
y = dataset.target
# print(x)
# print(x.shape) # (506, 13)
# print(y)
# print(y.shape) # (506,)
x_trn, x_tst, y_trn, y_tst = train_test_split(x, y,
                                              test_size=0.15,
                                              shuffle=True,
                                              random_state=42)

#2. 모델구성
model = Sequential()
model.add(Dense(64,input_dim = 13))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(1))

epochs = 5000

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_trn, y_trn, epochs = epochs, batch_size = 10)

#4. 평가,예측
loss = model.evaluate(x_tst, y_tst)
results = model.predict(x)
r2 = r2_score(y, results)

print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print("loss :", loss)
print('r2 스코어 :', r2)

# [실습] R2 기준 0.75 이상
# train_size : 0.80 / epochs : 300 / batch_size : 1 / hidden_layer : 13-64-64-64-64-1 / random_state : 55
# loss : 26.084592819213867
# r2 스코어 : 0.7043421792815056

# train_size : 0.85 / epochs : 300 / batch_size : 1 / hidden_layer : 13-64-64-64-64-1 / random_state : 55
# loss : 14.195486068725586
# r2 스코어 : 0.7032156561146404

# train_size : 0.85 / epochs : 400 / batch_size : 1 / hidden_layer : 13-64-64-64-64-1 / random_state : 55
# loss : 14.680927276611328
# r2 스코어 : 0.7188630108706033

# train_size : 0.85 / epochs : 600 / batch_size : 1 / hidden_layer : 13-64-64-64-64-1 / random_state : 55
                    # :: 1000            
# loss : 13.745532989501953
# r2 스코어 : 0.7260075780541853

# train_size : 0.85 / epochs : 600 / batch_size : 1 / hidden_layer : 13-64-64-64-64-1 / random_state : 55
                    # :: 1000        :: 10            :: layer 3~5
# loss : 14.206001281738281
# r2 스코어 : 0.7176477909792114

# train_size : 0.85 / epochs : 600 / batch_size : 1 / hidden_layer : 13-64-128-64-64-1 / random_state : 55
                    # :: 1000        :: 10            :: layer 3~5 ::13-64-128-128-64-1
# loss : 14.05739688873291
# r2 스코어 : 0.7203688092640023

# train_size : 0.85 / epochs : 600 / batch_size : 1 / hidden_layer : 13-64-256-128-64-1 / random_state : 42
# loss : 15.470252990722656
# r2 스코어 : 0.7111053146996053

# train_size : 0.85 / epochs : 1000 / batch_size : 1 / hidden_layer : 13-64-128-128-64-1 / random_state : 42
# loss : 15.928576469421387
# r2 스코어 : 0.6675548736779306

# train_size : 0.85 / epochs : 1200 / batch_size : 2 / hidden_layer : 13-64-128-128-64-1 / random_state : 42
# loss : 14.434605598449707
# r2 스코어 : 0.7217742767709168 

# train_size : 0.85 / epochs : 1200 / batch_size : 5 / hidden_layer : 13-64-128-128-64-1 / random_state : 42
# loss : 13.741518020629883
# r2 스코어 : 0.7108311196630781

# train_size : 0.85 / epochs : 1200 / batch_size : 3 / hidden_layer : 13-64-128-128-64-1 / random_state : 42
# loss : 13.933738708496094
# r2 스코어 : 0.7252240190334657 ***********

# train_size : 0.85 / epochs : 2000 / batch_size : 3 / hidden_layer : 13-64-128-128-64-1 / random_state : 42
# loss : 13.697587013244629
# r2 스코어 : 0.723958889079432 **********

# train_size : 0.85 / epochs : 1200 / batch_size : 3 / hidden_layer : 13-64-128-128-64-1 / random_state : 42
# loss : 15.049635887145996
# r2 스코어 : 0.6819962810798812

# train_size : 0.85 / epochs : 5000 / batch_size : 10 / hidden_layer : 13-64-128-128-64-1 / random_state : 42
# loss : 13.456618309020996
# r2 스코어 : 0.7321122112201539