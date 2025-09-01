from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

#1. 데이터
dataset = load_diabetes()
x = dataset.data
y = dataset.target
# print(x)
# print(y)
# print(x.shape, y.shape) # (442, 10) (442,)

# [실습] R2 0.62 이상
x_trn, x_tst, y_trn, y_tst = train_test_split(x, y,
                                              train_size = 0.85,
                                              shuffle = True,
                                              random_state=6974)

#2. 모델구성
model = Sequential()
model.add(Dense(400, input_dim = 10))
model.add(Dense(400))
model.add(Dense(200))
model.add(Dense(200))
model.add(Dense(1))

epochs = 500

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer ='adam')
model.fit(x_trn, y_trn, epochs = epochs, batch_size=3)

#4. 평가, 예측
loss = model.evaluate(x_tst, y_tst)
results = model.predict(x_tst)
R2 = r2_score(y_tst,results)

print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
print("R2 :", R2)

# train_size : 0.85 / random_state : 55 :: 4점대 유지

# train_size : 0.85 / random_state : 9 / epochs : 500 / hidden_layer : 10 400 400 200 200 1 / batch_size : 3
# R2 : 0.5742604015262193

# train_size : 0.85 / random_state : 13 / epochs : 500 / hidden_layer : 10 400 400 200 200 1 / batch_size : 3
# R2 : 0.3801096516937572

# train_size : 0.85 / random_state : 6974 / epochs : 500 / hidden_layer : 10 400 400 200 200 1 / batch_size : 3
# R2 : 0.6191175928790672