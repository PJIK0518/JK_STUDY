# 61_11.copy
##########################################################################
#0. 준비
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, OneHotEncoder, StandardScaler
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
import datetime
import numpy as np
import pandas as pb
import time

RS = 42
##########################################################################
#1 데이터
DS = load_digits()

x = DS.data
y = DS.target

# print(x.shape)  (1797, 64)
# print(y.shape)  (1797,)
""" print(np.unique(y, return_counts=True))
(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))
"""

x_trn, x_tst, y_trn, y_tst = train_test_split(x, y,
                                              train_size=0.8,
                                              shuffle= True,
                                              random_state=RS,
                                              stratify=y)
##########################################################################
#2 모델구성
#####################################
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

M_LSV = LinearSVC(C = 0.3)
M_LGR = LogisticRegression()
M_DTF = DecisionTreeClassifier()
M_RFC = RandomForestClassifier()

ML_list = [M_LSV, M_LGR, M_DTF, M_RFC]

''' loss
0.09086531400680542
DO
0.07925168424844742
CNN
0.089523546397686
0.06869980692863464
LSTM
loss : 0.18476565182209015

Conv1D
loss : 0.09292933344841003
'''


##########################################################################
#3 컴파일, 훈련
for model in ML_list:
    model.fit(x_trn,y_trn)

##########################################################################
#4. 평가 예측
for model in ML_list:
    score = model.score(x_tst,y_tst)
    
    print(f'{model} : ', score)
    
# LinearSVC(C=0.3) :  0.95
# LogisticRegression() :  0.9583333333333334
# DecisionTreeClassifier() :  0.8111111111111111
# RandomForestClassifier() :  0.9694444444444444