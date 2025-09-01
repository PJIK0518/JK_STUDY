# KNN in Regressor
# 가장 가까운 값들의 y 값에 대한 평균
# 그닥 많이 쓰는 거는 아님
# Classifier와 다르게 k 값이 홀수일 필요가 없음

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
import numpy as np
import random

import matplotlib.pyplot as plt

seed = 123
random.seed(seed)
np.random.seed(seed)
    
#1 데이터
DS =load_breast_cancer()
x = DS.data
y = DS.target

print(x.shape, y.shape)

x_trn, x_tst, y_trn, y_tst = train_test_split(
    x, y,
    train_size=0.7,
    random_state=seed,
    stratify=y)

scaler = StandardScaler()
x_trn = scaler.fit_transform(x_trn)
x_tst = scaler.transform(x_tst)

#2 모델구성
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import r2_score, accuracy_score, f1_score
model = KNeighborsClassifier(n_neighbors=5)     # k_neighbors랑 같은 Parameter

model.fit(x_trn, y_trn)
print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ', model.__class__.__name__,'ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print('acc :', model.score(x_tst, y_tst))

y_prd = model.predict(x_tst)

score1 = accuracy_score(y_tst, y_prd)
score2 = f1_score(y_tst, y_prd)

print('ACC :',score1)
print('F1S :',score2)

# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ KNeighborsClassifier ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# acc : 0.9707602339181286
# ACC : 0.9707602339181286
# F1S : 0.9769585253456221