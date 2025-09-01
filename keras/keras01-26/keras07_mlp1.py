from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([[1,2,3,4,5],
              [6,7,8,9,10]])
y = np.array([1,2,3,4,5]) # 2개의 데이터(x)가 들어가서 1개의 결과(y)를 가져오는 데이터, 반대도 가능
# x = np.array([[1,6],[2,7],[3,8],[4,9],[5,10]]) : 사실 이렇게 해야 원하는 데이터가 됨...
x = np.transpose(x) # transpose : 데이터의 행과 열을 바꿔주는 명령어 /// 작은 데이터량은 되지만 커지면 잘 안 돌아갈 수 도,,,

print(x.shape) # (2, 5) > tanspose > (5, 2)
print(y.shape) # (5, )

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=2)) # dimention = column = 열 = feature = 특성
                                  # 행 무시 열 우선 /// 행: 데이터의 갯수, 열: 데이터의 종류(특성)
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

epochs = 100

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y, epochs = epochs, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
results = model.predict([[6,11]]) # (1, 2), 마찬가지로 예측하는 데이터도 열(종류)는 맞아야한다...!

print('#########################################################################')
print('loss =', loss)
print('[6, 11]의 예측값 =', results)



# loss = 2.223998708719005e-13
# [6, 11]의 예측값 = [[5.9999986]]
# epochs 수도 적고, layer도 크게 하지 않았는데 좋은 유의성 >> column 수가 증가해서! : 데이터 수, 종류가 많아지면 완성도가 증가
                                                            # 데이터 종류가 너무 많아 져도 완성도가 안 좋아 질수도...!