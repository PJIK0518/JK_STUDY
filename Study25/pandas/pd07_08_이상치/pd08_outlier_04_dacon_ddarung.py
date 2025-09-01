# m10_04.copy

# tst, val : 데이터가 아깝다...
# But. 데이터의 과적합을 막기 위해서 필요한 과정
#      But. tst, val에 중요도가 높은 애들이 있다면??
#       >> tst에 한 번 썼던 데이터를 trn에 넣고 기존 trn에서 다시 tst 선정
#       >> 데이터 1/n 로 나눠서 n 반복 : n_split (데이터가 많다면 n 수를 높여서 진행)
#       >> 데이터의 소실 없이 훈련 가능

import warnings

import pandas as pd

warnings.filterwarnings('ignore')


import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

#1.데이터
path = './Study25/_data/dacon/따릉이/'
DS = pd.read_csv(path + 'train.csv', index_col = 0) 
# print(train_csv) # (1459, 11)
#                  # But. column_0은 index >> index_col로 제거
#                  # (1459, 10)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0) 
# print(test_csv)  # (715, 9)

submission_csv = pd.read_csv(path + 'submission.csv', index_col = 0)
"""
# print(submission_csv)
#                  # (715, 1)
#                  # NaN : 결칙치

# print(train_csv.shape)      # (1459, 10)
# print(test_csv.shape)       # (715, 9)
# print(submission_csv.shape) # (715, 1)

# # pandas로 가져온 파일에 대한 기능
# print(train_csv.columns)    # Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#                             #        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#                             #        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#                             #       dtype='object')
#                             # >> feature 불러오기
# print(train_csv.info())     #  0   hour                    1459 non-null   int64
#                             #  1   hour_bef_temperature    1457 non-null   float64
#                             #  2   hour_bef_precipitation  1457 non-null   float64
#                             #  3   hour_bef_windspeed      1450 non-null   float64
#                             #  4   hour_bef_humidity       1457 non-null   float64
#                             #  5   hour_bef_visibility     1457 non-null   float64
#                             #  6   hour_bef_ozone          1383 non-null   float64
#                             #  7   hour_bef_pm10           1369 non-null   float64
#                             #  8   hour_bef_pm2.5          1342 non-null   float64
#                             #  9   count                   1459 non-null   float64
#                             # >> feature 마다 데이터 갯수 >> 결칙치에 대한 정보 확인, 제거 or 예측, 제거는 애매함
#                             #                                                     >> 제거 : 데이터가 부족한거는 완성도 하락으로 직결
#                             #                                                     >> 예측 : 시간순서나, 유도리 있게 가능하면 예측 후 모델 제작
# print(train_csv.describe()) # >> 데이터의 평균, 최소, 분위 등을 제공
#               hour  hour_bef_temperature  hour_bef_precipitation  hour_bef_windspeed  ...  hour_bef_ozone  hour_bef_pm10  hour_bef_pm2.5        count
# count  1459.000000           1457.000000             1457.000000         1450.000000  ...     1383.000000    1369.000000     1342.000000  1459.000000      
# mean     11.493489             16.717433                0.031572            2.479034  ...        0.039149      57.168736       30.327124   108.563400      
# std       6.922790              5.239150                0.174917            1.378265  ...        0.019509      31.771019       14.713252    82.631733      
# min       0.000000              3.100000                0.000000            0.000000  ...        0.003000       9.000000        8.000000     1.000000      
# 25%       5.500000             12.800000                0.000000            1.400000  ...        0.025500      36.000000       20.000000    37.000000      
# 50%      11.000000             16.600000                0.000000            2.300000  ...        0.039000      51.000000       26.000000    96.000000      
# 75%      17.500000             20.100000                0.000000            3.400000  ...        0.052000      69.000000       37.000000   150.000000      
# max      23.000000             30.000000                1.000000            8.000000  ...        0.125000     269.000000       90.000000   431.000000      
# [8 rows x 10 columns]

## 결측치 처리 1. 삭제 ##
# print(train_csv.info())         # 데이터의 개수 및 특성 출력
# print(train_csv.isnull().sum()) # null 값의 모든 합을 출력
# print(train_csv.isna().sum())   # null 값의 모든 합을 출력
#     # hour                        0
#     # hour_bef_temperature        2
#     # hour_bef_precipitation      2
#     # hour_bef_windspeed          9
#     # hour_bef_humidity           2
#     # hour_bef_visibility         2
#     # hour_bef_ozone             76
#     # hour_bef_pm10              90
#     # hour_bef_pm2.5            117
#     # count                       0
# train_csv = train_csv.dropna()  # 데이터의 결측치를 삭제하고 덮어 씌워라
# print(train_csv.isnull().sum())
#     # hour                      0
#     # hour_bef_temperature      0
#     # hour_bef_precipitation    0
#     # hour_bef_windspeed        0
#     # hour_bef_humidity         0
#     # hour_bef_visibility       0
#     # hour_bef_ozone            0
#     # hour_bef_pm10             0
#     # hour_bef_pm2.5            0
#     # count                     0
# print(train_csv.info())
#     #  0   hour                    1328 non-null   int64
#     #  1   hour_bef_temperature    1328 non-null   float64
#     #  2   hour_bef_precipitation  1328 non-null   float64
#     #  3   hour_bef_windspeed      1328 non-null   float64
#     #  4   hour_bef_humidity       1328 non-null   float64
#     #  5   hour_bef_visibility     1328 non-null   float64
#     #  6   hour_bef_ozone          1328 non-null   float64
#     #  7   hour_bef_pm10           1328 non-null   float64
#     #  8   hour_bef_pm2.5          1328 non-null   float64
#     #  9   count                   1328 non-null   float64
# print(train_csv)
#     # (1328, 10)
"""

# ## 결측치 처리 2. 평균 ## 
DS = DS.fillna(DS.mean())
'''print(train_csv.isnull().sum())
print(train_csv.info())
'''

## 테스트의 결측치는? ## >> 제거는 절대XXXX >> 제출은 해야하니까
'''print(test_csv.info())
    #  #   Column                  Non-Null Count  Dtype
    # ---  ------                  --------------  -----
    #  0   hour                    715 non-null    int64
    #  1   hour_bef_temperature    714 non-null    float64
    #  2   hour_bef_precipitation  714 non-null    float64
    #  3   hour_bef_windspeed      714 non-null    float64
    #  4   hour_bef_humidity       714 non-null    float64
    #  5   hour_bef_visibility     714 non-null    float64
    #  6   hour_bef_ozone          680 non-null    float64
    #  7   hour_bef_pm10           678 non-null    float64
    #  8   hour_bef_pm2.5          679 non-null    float64
'''
test_csv = test_csv.fillna(test_csv.mean())
'''   # #   Column                  Non-Null Count  Dtype
    # ---  ------                  --------------  -----
    # 0   hour                    715 non-null    int64
    # 1   hour_bef_temperature    715 non-null    float64
    # 2   hour_bef_precipitation  715 non-null    float64
    # 3   hour_bef_windspeed      715 non-null    float64
    # 4   hour_bef_humidity       715 non-null    float64
    # 5   hour_bef_visibility     715 non-null    float64
    # 6   hour_bef_ozone          715 non-null    float64
    # 7   hour_bef_pm10           715 non-null    float64
    # 8   hour_bef_pm2.5          715 non-null    float64
print(test_csv.info())
'''

x = DS.drop(['count'], axis=1) # 앞에서 편집한 train_csv에서 count라는 axis=1 열만 짤라서 삭제
# print(x)        # (1459, 9)           # 참고로 axis = 0 은 행
y = DS['count']                # count 컬럼만 빼서 y로
# print(y.shape)  # (1459,)
x = x.values

x_trn, x_tst, y_trn, y_tst = train_test_split(
    x, y,
    train_size=0.9,
    shuffle=True,
    random_state=777
)
def outlier(data):
    out = []
    up = []
    low = []
    for i in range(data.shape[1]):
        col = data[:, i]
        Q1, Q3 = np.percentile(col, [25, 75])
        
        IQR = Q3 - Q1
        print('IQR :', IQR)
        
        lower_bound = Q1 - (1.5 * IQR)
        upper_bound = Q3 + (1.5 * IQR)
        
        out_i = np.where((col > upper_bound) | (col < lower_bound))[0]
        out.append(out_i)
        up.append(upper_bound)
        low.append(lower_bound)
    return out, up, low

OUT, UP, LOW = outlier(x)

print(OUT) # [array([ 0, 12]), array([6])]
print(UP)  # [19.0, 1200.0]
print(LOW) # [-5.0, -400.0]

import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, x.shape[1], figsize=(15, 9))

for i in range(x.shape[1]):
    axs[i].boxplot(x[:,i])
    axs[i].axhline(UP[i], color = 'red', label = 'upper_bound')
    axs[i].axhline(LOW[i], color = 'red', label = 'lower_bound')
    axs[i].legend()
    axs[i].set_title(f"Column {i}")
    
# plt.tight_layout()
# plt.show()
# exit()

from sklearn.preprocessing import RobustScaler
RSC = RobustScaler()
col = [2,3,6,7,8]

for i in col:
    x_trn_col = x_trn[:, i].reshape(-1, 1)
    x_tst_col = x_tst[:, i].reshape(-1, 1)
    
    RSC.fit(x_trn_col)
    x_trn[:, i] = RSC.transform(x_trn_col).reshape(-1)
    x_tst[:, i] = RSC.transform(x_tst_col).reshape(-1)


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

print('# Outlier 처리')
print('# R2 :',score1)
# Outlier 처리
# R2 : 0.6312666660308165

# PF R2 : 0.41282558888731413

# R2 : 0.42673463539841305
# R2 : 0.42673463539841305

# 최적 컬럼 : 3 개 
#  ['hour', 'hour_bef_temperature', 'hour_bef_precipitation']
# 삭제 컬럼 : 6 개 
#  ['hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility', 'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5']
# 최고 점수 75.646%