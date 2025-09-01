# 61-6.copy

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import time

#1. 데이터
DS = load_breast_cancer()
'''print(DS.DESCR) (569, 30)
sklearn에서는 DESCR 가능, pandas 에서는 descibe로!!
===================================== ====== ======
                                           Min    Max
    ===================================== ====== ======
    radius (mean):                        6.981  28.11
    texture (mean):                       9.71   39.28
    perimeter (mean):                     43.79  188.5
    area (mean):                          143.5  2501.0
    smoothness (mean):                    0.053  0.163
    compactness (mean):                   0.019  0.345
    concavity (mean):                     0.0    0.427
    concave points (mean):                0.0    0.201
    symmetry (mean):                      0.106  0.304
    fractal dimension (mean):             0.05   0.097
    radius (standard error):              0.112  2.873
    texture (standard error):             0.36   4.885
    perimeter (standard error):           0.757  21.98
    area (standard error):                6.802  542.2
    smoothness (standard error):          0.002  0.031
    compactness (standard error):         0.002  0.135
    concavity (standard error):           0.0    0.396
    concave points (standard error):      0.0    0.053
    symmetry (standard error):            0.008  0.079
    fractal dimension (standard error):   0.001  0.03
    radius (worst):                       7.93   36.04
    texture (worst):                      12.02  49.54
    perimeter (worst):                    50.41  251.2
    area (worst):                         185.2  4254.0
    smoothness (worst):                   0.071  0.223
    compactness (worst):                  0.027  1.058
    concavity (worst):                    0.0    1.252
    concave points (worst):               0.0    0.291
    symmetry (worst):                     0.156  0.664
    fractal dimension (worst):            0.055  0.208
    ===================================== ====== ======
'''
'''print(DS.feature_names)
['mean radius' 'mean texture' 'mean perimeter' 'mean area'
 'mean smoothness' 'mean compactness' 'mean concavity'
 'mean concave points' 'mean symmetry' 'mean fractal dimension'
 'radius error' 'texture error' 'perimeter error' 'area error'
 'smoothness error' 'compactness error' 'concavity error'
 'concave points error' 'symmetry error' 'fractal dimension error'
 'worst radius' 'worst texture' 'worst perimeter' 'worst area'
 'worst smoothness' 'worst compactness' 'worst concavity'
 'worst concave points' 'worst symmetry' 'worst fractal dimension']
'''
# print(type(DS)) <class 'sklearn.utils.Bunch'>

x = DS.data     # (569, 30)
y = DS.target   # (569,)
'''print(x.shape, y.shape) (569, 30) (569,)
'''
'''print(type(x)) <class 'numpy.ndarray'> 보통 입력값 출력값으로 빼냈을 때는 numpy
                                    #  > load_breast_cancer 자체는 dictionary 형태
                                    #   :key_value를 기준으로 데이터들을 묶어 놓은 형태
'''
'''print(x, y)
[[1.799e+01 1.038e+01 1.228e+02 ... 2.654e-01 4.601e-01 1.189e-01]
 [2.057e+01 1.777e+01 1.329e+02 ... 1.860e-01 2.750e-01 8.902e-02]
 [1.969e+01 2.125e+01 1.300e+02 ... 2.430e-01 3.613e-01 8.758e-02]
 ...
 [1.660e+01 2.808e+01 1.083e+02 ... 1.418e-01 2.218e-01 7.820e-02]
 [2.060e+01 2.933e+01 1.401e+02 ... 2.650e-01 4.087e-01 1.240e-01]
 [7.760e+00 2.454e+01 4.792e+01 ... 0.000e+00 2.871e-01 7.039e-02]]
 
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 1 0 0 0 0 0 0 0 0 1 0 1 1 1 1 1 0 0 1 0 0 1 1 1 1 0 1 0 0 1 1 1 1 0 1 0 0
 1 0 1 0 0 1 1 1 0 0 1 0 0 0 1 1 1 0 1 1 0 0 1 1 1 0 0 1 1 1 1 0 1 1 0 1 1
 1 1 1 1 1 1 0 0 0 1 0 0 1 1 1 0 0 1 0 1 0 0 1 0 0 1 1 0 1 1 0 1 1 1 1 0 1
 1 1 1 1 1 1 1 1 0 1 1 1 1 0 0 1 0 1 1 0 0 1 1 0 0 1 1 1 1 0 1 1 0 0 0 1 0
 1 0 1 1 1 0 1 1 0 0 1 0 0 0 0 1 0 0 0 1 0 1 0 1 1 0 1 0 0 0 0 1 1 0 0 1 1
 1 0 1 1 1 1 1 0 0 1 1 0 1 1 0 0 1 0 1 1 1 1 0 1 1 1 1 1 0 1 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 1 1 1 1 1 1 0 1 0 1 1 0 1 1 0 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1
 1 0 1 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 0 1 0 1 1 1 1 0 0 0 1 1
 1 1 0 1 0 1 0 1 1 1 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 1 0 0
 0 1 0 0 1 1 1 1 1 0 1 1 1 1 1 0 1 1 1 0 1 1 0 0 1 1 1 1 1 1 0 1 1 1 1 1 1
 1 0 1 1 1 1 1 0 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 0 1 0 1 1 1 1 1 0 1 1
 0 1 0 1 1 0 1 0 1 1 1 1 1 1 1 1 0 0 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 0 1
 1 1 1 1 1 1 0 1 0 1 1 0 1 1 1 1 1 0 0 1 0 1 0 1 1 1 1 1 0 1 1 0 1 0 1 0 0
 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 0 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 0 0 0 0 0 0 1]
'''

# 지금의 경우에는 데이터가 작아서 0 1 로만 된게 보이지만 더 많아지면 찾기 힘듦!

## numpy로 데이터 확인하는 방법
# print(np.unique(y, return_counts=True)) (array([0, 1]), array([212, 357], dtype=int64))
# unique_vals = np.unique(y)
# print(unique_vals) [0 1]


## pandas로 데이터 확인하는 방법
# print(pd.value_counts(y))
# 1    357
# 0    212

### pandas에서 호환되는 데이터 : Dataframe, Series
  # y는 numpy 형태의 데이터 이기 때문에 호환되는 형태로 전환이 필요함
# print(pd.DataFrame(y).value_counts())
# 1    357                                      
# 0    212

# print(pd.Series(y).value_counts())
# 1    357
# 0    212


x_trn, x_tst, y_trn, y_tst = train_test_split(x, y,
                                              train_size=0.7,
                                              shuffle= True,
                                              random_state=55,)
'''
print(x_trn.shape, x_tst.shape) (398, 30) (171, 30)
print(y_trn.shape, y_tst.shape) (398,) (171,)
'''

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
                                                        
''' loss
loss :  0.1797563135623932
DO
loss :  0.16818349063396454
CNN
loss :  0.39920616149902344
loss :  0.14133839309215546
LSTM
loss :  0.13542504608631134
Conv1D
loss :  0.16886097192764282
'''
                                                        
#3. 컴파일 훈련
# model.compile(loss='mse', optimizer='adam')               # mse는 결과값과 예측치의 차이를 수치화
                                                            # sigmoid에 의해서 0~1 사이의 값만 나옴
                                                            # mse는 이진분류에서는 무의미한 완성도 지표
for model in ML_list:
    model.fit(x_trn,y_trn)

#4. 평가, 예측
for model in ML_list:
    score = model.score(x_tst,y_tst)
    
    print(f'{model} : ', score)
    
# LinearSVC(C=0.3) :  0.9473684210526315
# LogisticRegression() :  0.9590643274853801
# DecisionTreeClassifier() :  0.9415204678362573
# RandomForestClassifier() :  0.9590643274853801