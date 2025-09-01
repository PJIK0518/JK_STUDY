## 20250530 실습_1 [ACC = 1]
## 61-9.copy

from sklearn.datasets import load_wine

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder


import numpy as np
import pandas as pd
import time

#1. 데이터
DS = load_wine()

x = DS.data
y = DS.target

x_trn, x_tst, y_trn, y_tst = train_test_split(x, y,
                                              train_size = 0.7,
                                              shuffle = True,
                                              random_state = 4165,
                                            #   stratify=y
)

# trn, tst에 y의 라벨 값이 불균형하게 들어갈수도!!!!
# 특히 데이터가 치중된 경우 모델이 애매해짐 >> stratify = y : y를 전략적으로 각 데이터를 나눠라

# print(x, y)
'''print(pd.value_counts(y))
1    71
0    59
2    48
'''
# print(x.shape, y.shape) (178, 13) (178,)

from tensorflow.keras.layers import Conv2D, Flatten

#2. 모델구성
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier



M_LSV = LinearSVC(C = 0.3)
M_LGR = LogisticRegression()
M_DTF = DecisionTreeClassifier()
M_RFC = RandomForestClassifier()

ML_list = [M_LSV, M_LGR, M_DTF, M_RFC]        

'''loss
0.8734215497970581
DO
0.5326288938522339
CNN
0.253219336271286
LSTM
0.7279239296913147
Conv1D
0.4058389961719513
'''

#3. 컴파일 훈련
for model in ML_list:
    model.fit(x_trn,y_trn)

#4. 평가 예측
for model in ML_list:
    score = model.score(x_tst,y_tst)
    
    print(f'{model} : ', score)

# LinearSVC(C=0.3) :  0.9629629629629629
# LogisticRegression() :  0.9259259259259259
# DecisionTreeClassifier() :  0.8703703703703703
# RandomForestClassifier() :  0.9814814814814815