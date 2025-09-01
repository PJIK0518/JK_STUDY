# 14.copy

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import time
# 시간과 관련된 모듈 import

#1. 데이터
# x = np.array([1,2,3,4,5,6,7,8,9,10])
# y = np.array([1,2,3,4,5,6,7,8,9,10])
# print(x.shape) # (10,)
# print(y.shape) # (10,)

x_train = np.array(range(100))
y_train = np.array(range(100))
x_test = np.array([8,9,10])
y_test = np.array([8,9,10])

#2. 모델구성
model = Sequential()
model.add(Dense(16, input_dim=1))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
start_time = time.time()
        # 현재 시간을 반환
print(start_time)
model.fit(x_train, y_train,epochs = 10000, batch_size=1,
          verbose=3)
        # verboes 0 : 침묵
                # 1 : default
                # 2 : progress bar 제거
                # 3 : epochs 만 표기
                
end_time = time.time()
print('소요시간', end_time - start_time, '초')
    ## [실습1.] 1000 epochs 에서 verbose 0~3 의 시간 차이
    # verbose 0 37.6186044216156 초
    # verbose 1 47.954198122024536 초
    # verbose 2 37.94215130805969 초
    # verbose 3 38.871519804000854 초
    ## [실습2.] 1000 epochs / verbose 1 일 때 batch 1 32 128 의 시간 차이
    # batch 1   48.66806197166443 초
    # batch 32  3.051562547683716 초
    # batch 128 1.558499813079834 초
    ## [실습3.] 1000 epochs / verbose 3 일 때 batch 1 10 50 100 의 시간 차이
    # batch 1   37.67032074928284 초
    # batch 10  4.62886381149292 초
    # batch 50  1.5738115310668945 초
    # batch 100 1.138549566268921 초
    ## [실습3.] 10000 epochs / verbose 3 일 때 batch 1 10 50 100 의 시간 차이
    # batch 1   370.75523114204407 초
    # batch 10  44.22027802467346 초
    # batch 50  13.185872077941895 초
    # batch 100 9.754504203796387 초

exit()
#4. 평가, 예측
l = model.evaluate(x_test, y_test)
r = model.predict([11])

print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print('loss :', l)
print('11의 예측값 :',r)



# loss : 4.26378937845584e-06
# 11의 예측값 : [[10.997014]]