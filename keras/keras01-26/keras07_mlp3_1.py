from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([range(10), range(21, 31), range(201,211)])
y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [10,9,8,7,6,5,4,3,2,1]])
print(x)        #[[  0   1   2   3   4   5   6   7   8   9],
                # [ 21  22  23  24  25  26  27  28  29  30],
                # [201 202 203 204 205 206 207 208 209 210]]
print(x.shape)  # (3, 10)

print(y)        #[[ 1  2  3  4  5  6  7  8  9 10],
                # [10  9  8  7  6  5  4  3  2  1]]
print(y.shape)  # (2, 10)

x = x.T
y = y.T
print(x)        #[[  0  21 201],
                # [  1  22 202],
                # [  2  23 203],
                # [  3  24 204],
                # [  4  25 205],
                # [  5  26 206],
                # [  6  27 207],
                # [  7  28 208],
                # [  8  29 209],
                # [  9  30 210]]
print(x.shape)  # (10, 3)

print(y)        #[[ 1 10],
                # [ 2  9],
                # [ 3  8],
                # [ 4  7],
                # [ 5  6],
                # [ 6  5],
                # [ 7  4],
                # [ 8  3],
                # [ 9  2],
                # [10  1]]
print(y.shape)  # (10, 2)
                # 3가지 입력값 10개를 넣어서 2가지 출력값 10개를 얻는 모델

#2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=3))
model.add(Dense(16))
model.add(Dense(4))
model.add(Dense(2))

epochs = 100
n = [[10, 31, 211], [11, 32, 212]]

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y, epochs = epochs, batch_size = 1)

#4. 평가, 예측
loss = model.evaluate(x, y)
results = model.predict(n)

print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print('loss :', loss)
print(n,'의 예측값 :', results)



# loss : 0.0001775445998646319
# [[10, 31, 211], [11, 32, 212]] 의 예측값 : [[ 1.1012095e+01  9.1687776e-03], [ 1.2012317e+01 -9.8712164e-01]]