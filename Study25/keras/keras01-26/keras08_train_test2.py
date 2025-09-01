from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
# print(x.shape) # (10,)
# print(y.shape) # (10,)

# [실습] 넘파이 리스트 슬라이싱
x_train = x[:7] # =[0:7] 데이터의 처음부터 70% 까지
x_test = x[7:]  # =[7:10] 데이터의 70%부터 끝까지
print(x_train.shape, x_test.shape) # (7,) (3,)

y_train = y[:7]
y_test = y[7:]
print(y_train.shape, y_test.shape) # (7,) (3,)

print(x_train) # [1 2 3 4 5 6 7]
print(y_train) # [1 2 3 4 5 6 7]
print(x_test)  # [ 8  9 10]
print(y_test)  # [ 8  9 10]

# x_train = np.array([1,2,3,4,5,6,7])
# y_train = np.array([1,2,3,4,5,6,7])
# x_test = np.array([8,9,10])
# y_test = np.array([8,9,10])

# exit()
#2. 모델구성
model = Sequential()
model.add(Dense(16, input_dim=1))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train,epochs = 100, batch_size=1)

#4. 평가, 예측
l = model.evaluate(x_test, y_test)
r = model.predict([11])

print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print('loss :', l)
print('11의 예측값 :',r)



# loss : 4.26378937845584e-06
# 11의 예측값 : [[10.997014]]