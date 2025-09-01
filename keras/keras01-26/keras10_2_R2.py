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
r = model.predict(x_test)

print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print('loss :', l)
print('[x]의 결과값 :', r)

from sklearn.metrics import r2_score, mean_squared_error

# def RMSE(y_test, y_predict) :
#    return np.sqrt(mean_squared_error(y_test, y_predict))
# rmse = RMSE(y_test, r)
# print('RMSE :', rmse)

r2 = r2_score(y_test, r)
print('r2 스코어 :', r2)



# loss : 3.3282344341278076
# [x]의 결과값 : [[ 9.253285  ]
                # [ 0.75868285]
                # [11.140974  ]
                # [ 1.7025274 ]
                # [ 5.4779067 ]
                # [ 8.309441  ]]
# r2 스코어 : 0.7659835815429688