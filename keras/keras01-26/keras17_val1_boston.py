#11_1.copy

import sklearn as sk
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np

#1. 데이터
dataset = load_boston()    
x = dataset.data
y = dataset.target
# print(x)
# print(x.shape) # (506, 13)
# print(y)
# print(y.shape) # (506,)
x_trn, x_tst, y_trn, y_tst = train_test_split(x, y,
                                              test_size=0.15,
                                              shuffle=True,
                                              random_state=2)

#2. 모델구성
model = Sequential()
model.add(Dense(64,input_dim = 13))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(1))

epochs = 1200

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_trn, y_trn, epochs = epochs, batch_size = 3,
          verbose=2,
          validation_split=0.2)

#4. 평가,예측
loss = model.evaluate(x_tst, y_tst)
results = model.predict(x)
r2 = r2_score(y, results)

print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print("loss :", loss)
print('r2 스코어 :', r2)

##[기존 최고치]
# train_size : 0.85 / epochs : 1200 / batch_size : 3 / hidden_layer : 13-64-128-128-64-1 / random_state : 42
# loss : 13.933738708496094
# r2 스코어 : 0.7252240190334657 ***********

''' validation 적용
train_size : 0.85 / epochs : 1200 / batch_size : 3 / hidden_layer : 13-64-128-128-64-1 / random_state : 42
loss : 26.947917938232422
r2 스코어 : 0.5879988937247986

loss : 24.593360900878906
r2 스코어 : 0.6985541922493144
'''