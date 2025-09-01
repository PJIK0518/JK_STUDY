from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRFRegressor
import pandas as pd
import numpy as np
import random

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

seed = 123
random.seed(seed)
np.random.seed(seed)
    
#1 데이터
DS =load_iris()
x = DS.data
y = DS.target

print(x.shape, y.shape)

x_trn, x_tst, y_trn, y_tst = train_test_split(
    x, y,
    train_size=0.7,
    random_state=seed,
    stratify=y)

#2 모델구성
# model1 = DecisionTreeClassifier(random_state=seed)
# model2 = RandomForestClassifier(random_state=seed)
# model3 = GradientBoostingClassifier(random_state=seed)
model = XGBClassifier(random_state=seed)

model.fit(x_trn, y_trn)
print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ', model.__class__.__name__,'ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print('acc :', model.score(x_tst, y_tst))
print(model.feature_importances_)
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ XGBClassifier ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# acc : 0.9555555555555556
# [0.03295855 0.02776272 0.75007254 0.18920612]

print(np.percentile(model.feature_importances_, 25)) # 0.03165959473699331
                                                     # np.percentile(A, n) : A의 n% 지점을 계산
                                                     # 0.75007254 : 100%
                                                             #    :  75%
                                                     # 0.18920612
                                                             #    :  50%
                                                     # 0.03295855 
                                                             #    :  25% > 보간법을 활용하여 계산
                                                     # 0.02776272 :   0%
                                                                     #   > if 문을 통해서 컬럼 삭제 및 선택 가능
                                                                     
CPT = np.percentile(model.feature_importances_, 25)
# print(type(CPT)) <class 'numpy.float64'>
COL_name = []

for i, FI in enumerate(model.feature_importances_):
    if FI <= CPT:
        COL_name.append(DS.feature_names[i])
    else:
        continue

# print(COL_name) ['sepal width (cm)']

x = pd.DataFrame(x, columns=DS.feature_names)
x = x.drop(columns=COL_name)

# print(x) [150 rows x 3 columns]
#      sepal length (cm)  petal length (cm)  petal width (cm)
# 0                  5.1                1.4               0.2
# 1                  4.9                1.4               0.2
# 2                  4.7                1.3               0.2
# 3                  4.6                1.5               0.2
# 4                  5.0                1.4               0.2
# ..                 ...                ...               ...
# 145                6.7                5.2               2.3
# 146                6.3                5.0               1.9
# 147                6.5                5.2               2.0
# 148                6.2                5.4               2.3
# 149                5.9                5.1               1.8

x_trn, x_tst, y_trn, y_tst = train_test_split(
    x, y,
    train_size=0.7,
    random_state=seed,
    stratify=y)

model.fit(x_trn, y_trn)

score = model.score(x_trn, y_trn)

print('acc_drop :', score)