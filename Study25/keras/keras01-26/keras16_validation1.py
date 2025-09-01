# 8-1.copy

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
# x = np.array([1,2,3,4,5,6,7,8,9,10])
# y = np.array([1,2,3,4,5,6,7,8,9,10])
# print(x.shape) # (10,)
# print(y.shape) # (10,)

## train set
x_train = np.array([1,2,3,4,5,6])
y_train = np.array([1,2,3,4,5,6])

x_val = np.array([7,8])
y_val = np.array([7,8])

## test set
x_test = np.array([9,10])
y_test = np.array([9,10])

#2. 모델구성
model = Sequential()
model.add(Dense(16, input_dim=1))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train,
          epochs = 100, batch_size=1,
          validation_date=(x_val,y_val))

#4. 평가, 예측
l = model.evaluate(x_test, y_test)
r = model.predict([11])

print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print('loss :', l)
print('11의 예측값 :',r)



# loss : 4.26378937845584e-06
# 11의 예측값 : [[10.997014]]