from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

#2. 모델구성 : 깜지의 그림 참조
model = Sequential()
model.add(Dense(4, input_dim=1)) # input : 넣어주는 데이터
model.add(Dense(16, input_dim=4)) # hyperparameter tuning : hidden layer에서 얼마나 많은 계산을 할지는 개발자가 조절해서
model.add(Dense(4, input_dim=16))                         # 최적의 수치를 뽑아내면 된다...!
model.add(Dense(1, input_dim=4)) # output : 원하는 결과

epochs = 300
    # epochs는 300으로 건드리지말고, hyperparameter tuning을 통해서...!
    # hyperparameter tunig을 통해서 연결되는 layer 수를 늘리거나 줄였을 때 왜 더 좋아지는지는 설명하기 매우 까다로움...!

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y, epochs = epochs)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('#############################################################################')
print('epochs : ',epochs)
print('loss :', loss)

results = model.predict([6])
print('6의 예측값 : ', results)

# epochs :  300
# loss : 0.00018629743135534227
# 1/1 [==============================] - 0s 66ms/step
# 6의 예측값 :  [[5.9784904]]