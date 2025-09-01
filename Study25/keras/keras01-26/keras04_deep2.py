from tensorflow.keras.models import Sequential # model은 순서대로 구성되는 걸로 쓸꺼다
from tensorflow.keras.layers import Dense # layer는 빽빽하게 연결시킬 것이다
import numpy as np

# 에포 100 고정
# loss 0.32 미만으로

#1. 데이터
x = np.array([1,2,3,4,5,6]) # python 언어 팁, 대괄호 : 여러 개의 한 덩어리를 list화 시킨다는 의미
y = np.array([1,2,3,5,4,6])                         # 꼭 데이터가 아니더라도 2개 이상은 list화

#2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim = 1)) 
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1)) # layer 순서대로 당연히 연결되는 input이라서 생략 가능

epochs = 100

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y, epochs = epochs)

#4. 평가, 예측
# n = 7
loss = model.evaluate(x, y)
# results = model.predict([n])

print('###################################################################################')
print('loss : ',loss)
print('epochs : ',epochs)
# print(n,'의 예측값 :', results)



# loss :  0.3239014446735382
# epochs :  100