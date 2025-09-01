# 11_2.copy

import sklearn as sk
print(sk.__version__)   # 1.1.3
import tensorflow as tf
print(tf.__version__)   # 2.9.3

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import numpy as np

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x)
print(y)        # 데이터를 찍어보고 소수점(회귀) or 정수 몇개로만 구성(분류) AI 모델 종류 결정
print(x.shape)  # (20640, 8)
print(y.shape)  # (20640,)
print(datasets.feature_names)
                # ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
x_trn, x_tst, y_trn, y_tst = train_test_split(x, y,
                                              test_size= 0.15,
                                              shuffle= True,
                                              random_state= 55)

#2. 모델구성
model = Sequential()
model.add(Dense(8,input_dim=8))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(1))
epochs = 5000

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_trn, y_trn, epochs = epochs, batch_size = 50,
          verbose=2,
          validation_split=0.2)

#4. 평가, 예측
loss = model.evaluate(x_tst, y_tst)
results = model.predict([x_tst])

from sklearn.metrics import r2_score, mean_squared_error
def RMSE(a, b) :
    return np.sqrt(mean_squared_error(a,b))

rmse = RMSE(y_tst, results)
R2 = r2_score(y_tst, results)

print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print('loss :', loss)
print('rmse :', rmse)
print('R2 :', R2)

##[기존]
# trn=0.85 / RS= 55 / e = 5000 / HL 8-64-128-128-64-64-1 / BS=50
# loss : 0.5563498139381409
# rmse : 0.7458885803295967
# R2 : 0.5843834853462511 ***********

##[갱신]
# loss : 0.5707632899284363
# rmse : 0.7554887341834895
# R2 : 0.5736160338881686