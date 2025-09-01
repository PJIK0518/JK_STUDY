from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import pandas as pd
import numpy as np
import random

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

seed = 123
random.seed(seed)
np.random.seed(seed)
    
#1 데이터
DS =load_diabetes()
x = DS.data
y = DS.target

print(x.shape, y.shape)

x_trn, x_tst, y_trn, y_tst = train_test_split(
    x, y,
    train_size=0.7,
    random_state=seed)

from sklearn.metrics import r2_score, accuracy_score, f1_score
from sklearn.preprocessing import PolynomialFeatures
PF = PolynomialFeatures(degree=2, include_bias=False)
x_trn = PF.fit_transform(x_trn)
x_tst = PF.transform(x_tst)

#2 모델구성
from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(x_trn, y_trn)
print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ', model.__class__.__name__,'ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print('R2 :', model.score(x_tst, y_tst))

y_prd = model.predict(x_tst)

score1 = r2_score(y_tst, y_prd)

print('R2 :',score1)

# PF R2 : 0.41282558888731413

# R2 : 0.32719165067422273
# R2 : 0.32719165067422273

# 최적 컬럼 : 3 개 
#  ['bmi', 's4', 's5']
# 삭제 컬럼 : 7 개 
#  ['age', 'sex', 'bp', 's1', 's2', 's3', 's6']
# 최고 점수 38.241%