from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

#2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim = 1)) 
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))

epochs = 100

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y, epochs = epochs, batch_size = 2)
    # batch : 데이터를 분할해서 훈련시킬 때 사용하는 단위
    # 1 epochs 당,,, 6개의 데이터를 한 번에 넣어서 하는게 아니라 1개씩 나눠서 훈련 후 취합
    # 기본적으로 loss 값도 낮아짐 > 통상적으로 그렇지만 데이터랑 모델에 따라 차이가 날 수 있음,,, 너무 큰 데이터에 너무 작은 batch 면 비효율적 일수도 
    # 메모리 부하도 감소
    # batch size가 데이터의 50% 가 넘으면 size에 맞게 했다가 나머지 진행

#4. 평가, 예측
# n = 7
loss = model.evaluate(x, y)
# results = model.predict([n])

print('###################################################################################')
print('loss : ',loss)
print('epochs : ',epochs)
# print(n,'의 예측값 :', results)



# loss :  0.3238629996776581
# epochs :  100