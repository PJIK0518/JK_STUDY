#0. AI 모델의 기본구성
#1. 데이터
#2. 모델구성
#3. 컴파일, 훈련
#4. 평가, 예측

import tensorflow as tf
print(tf.__version__)   # 2.9.3
import numpy as np
print(np.__version__)   # 1.21.1

    # from 하위 목록에서 improt 하위의 class를 가져오겠다
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터 : 우리가 준비해야 할 데이터, 보통 기본적으로 x, y, 지금 사용하는 데이터는 numpy 형태의 데이터
    # 전체적인 문법은 추후에 차차 배워갈꺼다...!
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

#2. 모델구성
model = Sequential()
    # model 종류
model.add(Dense(1, input_dim=1))
    # 모델 구성 : input_dim=1 1차원 값을 넣었을 때, Dense(1,) 1개를 뽑아내라 >> 이거를 원하는 성능이 나올 때까지 계속해서 반복

#3. 컴파일, 훈련
    # 컴파일 : 컴퓨터가 알아들을 수 있게 모델을 만들어가는 과정
model.compile(loss='mse', optimizer='adam')
    # loss, 최소 손실을 설정 / optimizer, 최적화... adam은 일단 성능 괜찮은 애 중에 하나라고 이해..!
model.fit(x, y, epochs=5000)
    # fit, (데이터를) 훈련시키다 or 맞추게하다 / epochs, 반복 횟수 (많이 시키면 정확도 상승)
    # fit을 하면 할 수록 가중치 발생 > 최적의 가중치를 찾는게 AI 개발

#4. 평가, 예측
result = model.predict(np.array([5]))
print('5의 예측값 : ', result)

    # "데이터 > 모델구성 > 컴파일, 훈련 > 평가, 예측" 네 단계로 모든 코딩이 진행된다..!
    # x, y 값 (데이터)을 늘려줬을 때도 가중치는 상승
    # 모델은 완전히 랜덤하게 들어가기 때문에 매번 오차가 다르게 발생 가능...!