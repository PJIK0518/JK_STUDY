from tensorflow.keras.models import Sequential
    # from keras.models import Sequential
    # 보통 노란줄이나 빨간줄이 끄어지면 문제 / 노란줄의 경우 버전 문제로 인식 불가한 상황 / 빨간줄은 없는 명령어?
    # 1,2 번 줄이 다르지만 버전 문제 정도의 사소한 문제...! 1번줄도 인식이 가능한거
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,4,5,6])
    # 개발자는 조건문(if) + 반복문(co)만 알면 언어에 따라(python, C++ 등등) 문법만 익히면 된다

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=8000)
    # 100번째 가중치를 가져온다 (100번 중에 최고 좋은 건 X)

print('############################################################################')

#4. 평가, 예측
loss = model.evaluate(x, y)
print('로스 : ', loss)
    # x, y 값을 넣어서 평가한다 >> 최종 가중치로 계산한 loss 값
results = model.predict([1,2,3,4,5,6,7])
print('예측값 : ', results)
    # results의 경우 여러개의 값을 산출할 수 있음