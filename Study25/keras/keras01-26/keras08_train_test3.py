from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split 

import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
# print(x.shape) # (10,)
# print(y.shape) # (10,)

# [실습] 리스트 슬라이싱

x_train, x_test, y_train, y_test = train_test_split(x, y,               # 순서대로 train, test로 나눔
                                                    train_size = 0.7, 
                                                    test_size = 0.3,    # 둘 중 하나 생략가능 / 둘 다 썼는데 1 안되면 손실 값 발생 / default : 0.25
                                                    shuffle = True,     # 생략 가능 / default : True
                                                                        # True : 렌덤하게 섞어서 뽑기 / 통상적으로 True가 완성도가 더 높음
                                                                        # False : 안 섞고 뽑기 >> x[0:7]랑 같은 의미 / 경우에 따라 필요하기도
                                                                                # 데이터 열에 시간이 들어가면 False로 자르고, True로 한 번 더 하기도
                                                                                # ex) 날씨 주가 코인
                                                    random_state = 518) # 무작위 난수, 무작위 난수표에 따라 숫자를 뽑아옴, 생략가능
                                                                        # 무작위로 뽑은 데이터를 고정하는 역할, 생략시 계속 바뀜
print(x_train.shape, x_test.shape) # (7,) (3,)
print(y_train.shape, y_test.shape) # (7,) (3,)
print(x_train) # [2 7 4 9 5 6 10]
print(y_train) # [2 7 4 9 5 6 10]
print(x_test)  # [1 3 8]
print(y_test)  # [1 3 8]

exit()

#2. 모델구성
model = Sequential()
model.add(Dense(32, input_dim=1))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train,epochs = 100, batch_size=1)

#4. 평가, 예측
l = model.evaluate(x_test, y_test)
r = model.predict([11])

print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print('loss :', l)
print('11의 예측값 :',r)



# loss : 1.6678095571265317e-09
# 11의 예측값 : [[10.999933]]