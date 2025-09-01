from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13, 8,14,15, 9, 6,17,23,21,20])
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size = 0.3,
                                                    random_state = 333)

print(x.shape)
print(y.shape)

#2. 모델구성
model = Sequential()
model.add(Dense(32, input_dim = 1))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

epochs = 100

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = epochs, batch_size=2)

#4. 평가, 예측
l = model.evaluate(x_test, y_test)
r = model.predict([x])

print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print('loss :', l)
print('[x]의 결과값 :', r)

import matplotlib.pyplot as plt # 그래프 그리려면 x, y 축 데이터 양식 통일 필요
plt.scatter(x, y) 
plt.plot(x, r, color = 'red')
plt.show()



# loss : 3.5507261753082275
# [x]의 결과값 : [[ 0.7717023]
                # [ 1.769427 ]
                # [ 2.7671506]
                # [ 3.7648754]
                # [ 4.7626014]
                # [ 5.760323 ]
                # [ 6.7580485]
                # [ 7.755773 ]
                # [ 8.753497 ]
                # [ 9.751222 ]
                # [10.748946 ]
                # [11.746667 ]
                # [12.744392 ]
                # [13.742119 ]
                # [14.739842 ]
                # [15.737568 ]
                # [16.735292 ]
                # [17.733017 ]
                # [18.730742 ]
                # [19.728466 ]]