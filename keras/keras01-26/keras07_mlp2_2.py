from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array(range(10))     # range(n) : 0 ~ n-1
print(x)                    # [0 1 2 3 4 5 6 7 8 9]
print(x.shape)              # (10,)
                            # 같은 내용을 여러번 정의 하면 최종적으로 한 걸로 덮어짐...!
x = np.array(range(1, 10))  # range(n, m) : n ~ m-1
print(x)                    # [1 2 3 4 5 6 7 8 9]
print(x.shape)              # (9,)

x = np.array(range(1, 11))
print(x)                    # [1  2  3  4  5  6  7  8  9 10]
print(x.shape)              # (10,)

# x = np.array(range(10), range(21, 31), range(201, 211))
# print(x)                  # error :array() takes from 1 to 2 positional arguments but 3 were given
# print(x.shape)            # 3가지의 범위를 한번에 인식 못하는 경우! >> 2개 이상은 list

x = np.array([range(10), range(21, 31), range(201, 211)])
print(x)                    # [[  0   1   2   3   4   5   6   7   8   9]
                            #  [ 21  22  23  24  25  26  27  28  29  30]
                            #  [201 202 203 204 205 206 207 208 209 210]]
print(x.shape)              # (3, 10) : 잘못된 데이터일 가능성...!
x = x.T                     # x = np.transpose(x) : 전치행렬과 같은 역할 
print(x)                    # [[  0  21 201]
                            #  [  1  22 202]
                            #  [  2  23 203]
                            #  [  3  24 204]
                            #  [  4  25 205]
                            #  [  5  26 206]
                            #  [  6  27 207]
                            #  [  7  28 208]
                            #  [  8  29 209]
                            #  [  9  30 210]]
print(x.shape)              # (10, 3)

y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])    
    # 실습 >> [10, 31, 211] 예측

#2. 모델구성
model = Sequential()
model.add(Dense(16, input_dim = 3))
model.add(Dense(10))        # 보통 항아리 형태로 hidden layer를 구성
model.add(Dense(10))        # [3 > 1 > 10 > 5 > 3 > 1] 등의 형태로 가면 데이터가 소실되거나 변형 될 수 있다
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs = 1000, batch_size=2)

#4. 평가, 예측
n = [[10, 31, 211]]
loss = model.evaluate(x, y)
results = model.predict(n)

print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print(n, '의 예측값 :', results)
print('loss :', loss)



# [[10, 31, 211]] 의 예측값 : [[10.999982]]
# loss : 1.2005330063402653e-10