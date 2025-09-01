## keras20_sigmoid_metrics_cancer.c0py

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
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

#2. 모델구성

path_MCP = './_save/keras28_mcp/06_cancer/'
model = load_model(path_MCP + 'keras28_0604_1425_0011-0.2163.h5')

'''
[MCP]
acc_score :  0.9064327485380117
소요시간 :  0.98831 초
[load]
acc_score :  0.9064327478408813
'''

# model = Sequential()
# model.add(Dense(64, input_dim=30, activation= 'relu'))  # activation = 'relu' : 음수는 0으로
# model.add(Dense(64, activation= 'relu'))
# model.add(Dense(64, activation= 'relu'))
# model.add(Dense(64, activation= 'relu'))
# model.add(Dense(64, activation= 'relu'))
# model.add(Dense(1, activation= 'sigmoid'))              # 결과가 -무한대에서 무한대까지 나올 수 있음
                                                        # 우리가 필요한 건 0 or 1로 나누기
                                                        # 0.5를 기준으로 왼쪽(작은건) > 0
                                                                      # 오른쪽(큰건) > 1
                                                        # 1단계 : 이진분류는 무조건 Sigmoid
                                                        # activation = 'sigmoid' : 0~1로 결과값 한정
                                                        # 마지막 노드는 1 만 나올 수 있다
                                                        
#3. 컴파일 훈련
# model.compile(loss='mse', optimizer='adam')               # mse는 결과값과 예측치의 차이를 수치화
                                                            # sigmoid에 의해서 0~1 사이의 값만 나옴
                                                            # mse는 이진분류에서는 무의미한 완성도 지표

# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics = ['accuracy'])                       
# # 2단계 : 이진분려는 무조건 binary_crossentropy
# # 수식 확인!
# # 1) 시그마 안의 수식이 예측이 맞으면 log1,
# #                      틀리면 log0로 계산됨
# # 2) log1 = 0 / log0 = -(무한대)
# # 3) 마지막에 n 빵치고 마이너스
# # 4) 틀린 예측이 많아지면 숫자가 커짐
# #    = 작을수록 좋다!!!!

# ES = EarlyStopping(monitor= 'val_loss',
#                    mode= 'min',
#                    patience= 10,
#                    restore_best_weights=True)

# ################################# mpc 세이브 파일명 만들기 #################################
# ### 월일시 넣기
# import datetime

# path_MCP = './_save/keras28_mcp/06_cancer/'

# date = datetime.datetime.now()
# # print(date)            
# # print(type(date))       
# date = date.strftime('%m%d_%H%M')              

# # print(date)             
# # print(type(date))

# filename = '{epoch:04d}-{val_loss:.4f}.h5'
# filepath = "".join([path_MCP,'keras28_',date, '_', filename])

# MCP = ModelCheckpoint(monitor='val_loss',
#                       mode='auto',
#                       save_best_only=True,
#                       filepath= filepath # 확장자의 경우 h5랑 같음
#                                          # patience 만큼 지나기전 최저 갱신 지점        
#                       )

# S_time = time.time()
# hist = model.fit(x_trn, y_trn, epochs=10000, batch_size=100,
#                  verbose=2,
#                  validation_split=0.2,
#                  callbacks=[ES,MCP])
# E_time = time.time()

#4. 평가, 예측
results = model.evaluate(x_tst, y_tst)
y_pred = model.predict(x_tst)

# print(results)                          # [loss, accuracy] : [0.24983803927898407, 0.9122806787490845]
# print("loss : ", results[0])            # loss :  0.17432253062725067
# print("acc  : ", results[1])            # acc  :  0.9181286692619324
# print("acc  : ", round(results[1], 5))  # acc  :  0.91813
# print("acc  : ", round(results[1], 4))  # acc  :  0.9181
# print(y_pred)                           # 실제로 0, 1의 값은 아님 : but. acc = 0.9181? 

''' print(y_pred[:10])
[[3.8931668e-03]
 [9.7395158e-01]
 [9.7995555e-01]
 [4.2187333e-02]
 [9.9937987e-01]
 [1.3294816e-04]
 [9.8592103e-01]
 [5.4251254e-03]
 [6.8896967e-01]
 [3.5369903e-02]]
'''
# y_pred = (np.round(y_pred))
'''print(y_pred[:10])
[[0.]
 [1.]
 [1.]
 [0.]
 [1.]
 [0.]
 [1.]
 [0.]
 [1.]
 [0.]]
'''

from sklearn.metrics import accuracy_score

# accuracy_score = accuracy_score(y_tst,y_pred)
# 변수와 함수 이름이 같아도 정의 가능
# ValueError: Classification metrics can't handle a mix of binary and continuous targets
# 이진법과 실수를 accuracy_score 함수에 넣을 수 없다
# metrics의 acc에서 이미 이진법으로 바꿔서 계산해준 것

# Time = E_time - S_time

print("acc_score : ", results[1])      # acc_score :  0.9005847953216374
# print("소요시간 : ", round(Time, 5), "초")  # 소요시간 :  2.28037 초

### [실습] acc > 0.97

''' tuning
'''