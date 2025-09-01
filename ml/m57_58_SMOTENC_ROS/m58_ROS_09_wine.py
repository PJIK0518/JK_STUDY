## 기본적인 SMOTE 알고리즘 :: 보간법!!

import random
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import load_wine
from keras.models import Sequential
from keras.layers import Dense


RS = 518
random.seed(RS)
np.random.seed(RS)
tf.random.set_seed(RS)

## 모델의 초기 Weight 값을 seed 값을 기준으로 고정!
## 통상적으로 weight 값은 아주 작은 값으로 무작위로 배정

#1. 데이터

DS = load_wine()

x = DS.data
y = DS.target

""" 데이터 정보
print(x.shape)(178, 13)
print(y.shape)(178,)

print(np.unique(y, return_counts=True))
(array([0, 1, 2]), array([59, 71, 48])) :: 실질적으로 데이터를 증폭할 필요 없음!
                                        :: 필요한 경우 : 70~80%의 데이터가 편향된 경우
                                        
print(y)
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]
 
print(pd.value_counts(y))
1    71
0    59
2    48
"""

#################### [실습] ####################
# y = label 2 가 부족한 편향 데이터를 만들어보자!
x_bias = x[:-40]
y_bias = y[:-40]
"""
print(x_bias.shape)(138, 13)
print(y_bias.shape)(138,)
print(np.unique(y_bias, return_counts=True)) (array([0, 1, 2]), array([59, 71,  8]))
"""
################################################
# 증폭을 통해서 label 0 1 2 의 수를 비슷하게
# How?? KNN algorithm : K-Nearest Neighbor 가까운 이웃
#                       K : 비교하는 최고 가까운 데이터의 수(임의의 상수)
#       가까운 columns 값들이 가지는 y label을 기준으로 새로운 데이터
#       y label을 기준 >> 지도학습!

#   ex) 임의의 데이터를 생성해서 KNN algorithm을 적용하면
#       K = k일때 가장 가까운 k개의 데이터 중에서 y label이 많은 값으로 데이터 생성
#       가까운 거리 : 유클리드 거리(피타고라스) 기준
#  but. 컬럼수, 데이터수가 많아지면 거리 계산하는 것에 연산량이 증가!
#       통상적으로 K = 3 or 5 정도(홀수형태로, 뭐가 많은지 비교해야하니까)

#  순서 : train_test_split 이 후에 SMOTE
#         증폭되지 않은 원 값을 통해 평가

x_trn, x_tst, y_trn, y_tst =train_test_split(
    x_bias, y_bias, train_size=0.75,
    shuffle=True, stratify=y_bias,
    random_state=RS
)

################################################
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
# ROS : 편향데이터를 단순 복사
ROS = RandomOverSampler(random_state=RS,
                        sampling_strategy={0:50, 2:33})
# smote = SMOTE(random_state=RS,
#               k_neighbors=5,                    # 몇 개의 값을 기준으로 가까운 애들을 알아볼 것인가? Default 5
#             # sampling_strategy={0:50, 2:33},   # 몇 개까지 증폭시킬것이냐 Default 'auto' : 최대 label로
#                                                 # 가장 많은 Label 값의 몇 개로 증폭시킬지 설정 가능 : ditionary 형태로...!
#             # n_jobs=-1                         # CPU 몇 스레드 쓸지 설정, 컬럼 및 데이터가 많아지면 오래걸림 : 0.12.0 이후로는 안써도 -1
#                                                 # TypeError: SMOTE.__init__() got an unexpected keyword argument 'n_jobs'
# )

# SMOTE : 연산량 계산
# 원 데이터
# x = (N, 10)
# y = {0 : 100,000, 1 : 30,000, 2 : 10,000}
# > 100,000 씩 증폭한다면?
# label 0 : 새로 찍히는 데이터 없음 : 0
# label 1 : 70,000개의 새로운 데이터 * 30,000개의 데이터와 거리 계산 * 10개의 컬럼 = 21000000000
# label 2 : 90,000개의 새로운 데이터 * 10,000개의 데이터와 거리 계산 * 10개의 컬럼 =  9000000000

# 연산량 줄이려면?
# 100,000 30,000 10,000 > 10*(10,000 3,000 1,000)으로 진행해서 연결
# 7,000 * 3,000 * 10 = 210000000
                #    21000000000
# 9,000 * 1,000 * 10 =  90000000
                #     9000000000
# > 1/10 이지만 한 번의 연산에 1/100로 연산량이 감소하고, 10번 계산해서 합산하면 1/10

x_trn, y_trn = ROS.fit_resample(x_trn, y_trn)

# exit()
""" 파라미터 조절
[k_neighbors=5, sampling_strategy='auto']
(array([0, 1, 2]), array([53, 53, 53]))
[k_neighbors=5, sampling_strategy={0:50, 2:33}]
(array([0, 1, 2]), array([50, 53, 33]))
"""

# Split 된 기준 가장 많은 label 수로 모든 label을 맞춤
# SMOTE : label 간의 불균형을 맞추는 목적
#      >> 기본적으로는 회귀에서 불가능
#      >> 계량시켜서 사용하는 애들도 있기는 함 > 필요한면 찾아봐


#2. 모델
model = Sequential()
model.add(Dense(10, input_shape = (13,)))
model.add(Dense(3, activation='softmax'))

#3. 컴파일 훈련
model.compile(loss = 'sparse_categorical_crossentropy',         ## 원핫이 되지않은 다중분류
              optimizer = 'adam',
              metrics = ['acc'])

model.fit(x_trn, y_trn,
          epochs = 50)
#4. 평가 예측
loss = model.evaluate(x_tst, y_tst)

print('loss :', loss[0])
print('accu :', loss[1])

y_prd = model.predict(x_tst)
print(y_prd)

y_prd = np.argmax(y_prd, axis=1)
print(y_prd)

accu = accuracy_score(y_tst, y_prd)
f1sc = f1_score(y_tst, y_prd, average='macro')  # 다중분류에서 그냥 F1 쓰면 뜨는 애러
                                                # ValueError: Target is multiclass but average='binary'.
                                                # Please choose another average setting, one of [None, 'micro', 'macro', 'weighted']
print('accu :', accu)
print('f1sc :',f1sc)

#################### [실습] ####################
#1. 원데이터
# accu : 0.6
# f1sc : 0.5188536953242836

#2. 편향데이터
# accu : 0.8857142857142857
# f1sc : 0.6073815751235107

#3. SMOTE 적용
# accu : 0.6
# f1sc : 0.453781512605042

#4. ROS
# accu : 0.6571428571428571
# f1sc : 0.48144499178981937