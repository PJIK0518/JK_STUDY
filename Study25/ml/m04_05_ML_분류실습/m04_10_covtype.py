## 20250530 실습_2 [ACC = 0.925]
# 61-10
# keras24_softmax3_fetch_covtype.copy

from sklearn.datasets import fetch_covtype

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, MaxPool2D, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

from sklearn.datasets import get_data_home
# print(get_data_home())

# exit()

# from imblearn.over_sampling import RandomOverSampler, SMOTE

RS = 55

#1. 데이터
DS = fetch_covtype()

x = DS.data
y = DS.target

# ros = RandomOverSampler(random_state=RS)
# x, y = ros.fit_resample(x, y)

# smt = SMOTE(random_state=RS)
# x, y = smt.fit_resample(x, y)

# print(x.shape)  (581012, 54)
# print(y.shape)  (581012,)
# print(np.unique(y, return_counts=True))
# array([0, 1, 2, 3, 4, 5, 6]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510]

x_trn, x_tst, y_trn, y_tst = train_test_split(x, y,
                                                train_size = 0.9,
                                                shuffle = True,
                                                random_state = RS,
                                                # stratify=y,
                                                )

# print(x.shape) (581012, 54)
# print(y.shape) (581012, 7)

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
0.5709723830223083
DO
0.7167189717292786
CNN
0.8257542252540588
0.7803897261619568
0.741298258304596
LSMT
3.2655792236328125
Conv1D
1.140297293663025
'''

#3. 컴파일 훈련
for model in ML_list:
    model.fit(x_trn,y_trn)

#4. 평가 예측
for model in ML_list:
    score = model.score(x_tst,y_tst)
    
    print(f'{model} : ', score)

# LinearSVC(C=0.3) :  0.7036074489690545
# LogisticRegression() :  0.6150218581115968
# DecisionTreeClassifier() :  0.9431344876252108
# RandomForestClassifier() :  0.9571787546039723