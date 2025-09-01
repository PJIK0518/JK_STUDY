from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([range(10)])
y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [10,9,8,7,6,5,4,3,2,1],
              [9,8,7,6,5,4,3,2,1,0]])
print(x)       # [[0 1 2 3 4 5 6 7 8 9]]
print(y.shape) # (3, 10)
x = x.T
y = y.T

#2. 모델구성
model = Sequential()
model.add(Dense(27, input_dim = 1))
model.add(Dense(18))
model.add(Dense(9))
model.add(Dense(3))
epochs = 100
n = [[10]]

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y, epochs = epochs, batch_size = 1)

#4. 평가, 예측
l = model.evaluate(x,y)
r = model.predict(n)

print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print('loss :', l)
print(n, '의 예측값 :', r)


# loss : 4.768955629702587e-12
# [[10]] 의 예측값 : [[ 1.1000000e+01  3.6805868e-06 -9.9999851e-01]]