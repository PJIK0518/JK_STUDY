import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,7,5,7,8,6,10])
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size = 0.3,
                                                    random_state = 333)

#2. 모델구성
model = Sequential()
model.add(Dense(16, input_dim = 1))
model.add(Dense(8))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
print ('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
loss = model.evaluate(x_test, y_test)
results = model.predict([x])

print('loss :', loss)
print('[x]의 예측값 :', results)

# 그래프 그리기
import matplotlib.pyplot as plt     # matplotlib : python 그래프 그리기 관련 library
plt.scatter(x, y)                   # 데이터 점 찍기
plt.plot(x, results, color = 'red') # 데이터 선 긋기
plt.show()



# loss : 0.5501615405082703
# [x]의 예측값 : [[1.8222167]
                # [2.6339364]
                # [3.4456558]
                # [4.2573757]
                # [5.069095 ]
                # [5.880815 ]
                # [6.6925344]
                # [7.5042534]
                # [8.315973 ]
                # [9.127692 ]]