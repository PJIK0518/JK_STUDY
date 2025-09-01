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

# 함수만드는 코드
def RMSE(y_test, y_predict) :
# def AAA(a,b) : a,b를 포함하는 함수 AAA 정의한다
# 함수니까 a와 b의 데이터 수는 같아야한다
    return np.sqrt(mean_squared_error(y_test, y_predict))
# return : 결과를 가져온다
# np.sqrt(mean_squared_error(a,b)) : a, b로 뽑은 mse에 root를 씌운다

rmse1 = RMSE(y_test, r)
rmse2 = np.sqrt(l)
print('RMSE :', rmse1)
print('대조값 :', rmse2)

# loss : 3.662158727645874
# [x]의 결과값 : [[ 9.8280115 ]
                # [ 0.68958676]
                # [11.858768  ]
                # [ 1.7049674 ]
                # [ 5.7664886 ]
                # [ 8.812633  ]]
# RMSE : 1.9136766317245753