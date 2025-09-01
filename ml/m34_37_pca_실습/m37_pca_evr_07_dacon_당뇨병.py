# m10_07.copy

# tst, val : 데이터가 아깝다...
# But. 데이터의 과적합을 막기 위해서 필요한 과정
#      But. tst, val에 중요도가 높은 애들이 있다면??
#       >> tst에 한 번 썼던 데이터를 trn에 넣고 기존 trn에서 다시 tst 선정
#       >> 데이터 1/n 로 나눠서 n 반복 : n_split (데이터가 많다면 n 수를 높여서 진행)
#       >> 데이터의 소실 없이 훈련 가능

import warnings

warnings.filterwarnings('ignore')


from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler

#1. 데이터
path = './Study25/_data/dacon/당뇨병/'

trn_csv = pd.read_csv(path + "train.csv", index_col=0)
'''print(trn_csv) [652 rows x 9 columns]
           Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  DiabetesPedigreeFunction  Age  Outcome
ID
TRAIN_000            4      103             60             33      192  24.0                     0.966   33        0
TRAIN_001           10      133             68              0        0  27.0                     0.245   36        0
TRAIN_002            4      112             78             40        0  39.4                     0.236   38        0
TRAIN_004            1      114             66             36      200  38.1                     0.289   21        0
TRAIN_647            1       91             64             24        0  29.2                     0.192   21        0
TRAIN_648           10      122             68              0        0  31.2                     0.258   41        0
TRAIN_649            8       84             74             31        0  38.3                     0.457   39        0
TRAIN_650            2       81             72             15       76  30.1                     0.547   25        0
TRAIN_651            1      107             68             19        0  26.5                     0.165   24        0
'''
'''print(trn_csv.info())
 #   Column                    Non-Null Count  Dtype
---  ------                    --------------  -----
 0   Pregnancies               652 non-null    int64
 1   Glucose                   652 non-null    int64
 2   BloodPressure             652 non-null    int64
 3   SkinThickness             652 non-null    int64
 4   Insulin                   652 non-null    int64
 5   BMI                       652 non-null    float64
 6   DiabetesPedigreeFunction  652 non-null    float64
 7   Age                       652 non-null    int64
 8   Outcome                   652 non-null    int64
'''

tst_csv = pd.read_csv(path + "test.csv", index_col=0)
'''print(tst_csv) [116 rows x 8 columns]
           ID  Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  DiabetesPedigreeFunction  Age
0    TEST_000            5      112             66              0        0  37.8                     0.261   41
1    TEST_001            3      107             62             13       48  22.9                     0.678   23
2    TEST_002            3      113             44             13        0  22.4                     0.140   22
4    TEST_004            1      107             72             30       82  30.8                     0.821   24
111  TEST_111           10      111             70             27        0  27.5                     0.141   40
112  TEST_112            1      119             54             13       50  22.3                     0.205   24
113  TEST_113            3      187             70             22      200  36.4                     0.408   36
114  TEST_114            3      100             68             23       81  31.6                     0.949   28
115  TEST_115            2       84              0              0        0   0.0                     0.304   21
'''
'''print(tst_csv.info())
 #   Column                    Non-Null Count  Dtype
---  ------                    --------------  -----
 0   Pregnancies               116 non-null    int64
 1   Glucose                   116 non-null    int64
 2   BloodPressure             116 non-null    int64
 3   SkinThickness             116 non-null    int64
 4   Insulin                   116 non-null    int64
 5   BMI                       116 non-null    float64
 6   DiabetesPedigreeFunction  116 non-null    float64
 7   Age                       116 non-null    int64
'''

sub_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)
''' print(sub_csv) [116 rows x 1 columns]
          Outcome
ID
TEST_000        0
TEST_001        0
TEST_002        0
TEST_003        0
TEST_004        0
...           ...
TEST_111        0
TEST_112        0
TEST_113        0
TEST_114        0
TEST_115        0
'''

## nan_null / 0이 nan 값, 대체!

x = trn_csv.drop(['Outcome'], axis=1)
y = trn_csv['Outcome']

x = x.replace(0, np.nan)
x = x.fillna(x.mean())
# print(x.info())

tst_csv = tst_csv.replace(0, np.nan)
tst_csv = tst_csv.fillna(tst_csv.mean())

C = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']

x_trn, x_tst, y_trn, y_tst = train_test_split(
    x, y,
    train_size=0.9,
    shuffle=True,
    random_state=777,
    stratify=y
)

from sklearn.decomposition import PCA
import time

pca = PCA(n_components=x_trn.shape[1])

x_trn = pca.fit_transform(x_trn)
x_tst = pca.fit_transform(x_tst)

evr = pca.explained_variance_ratio_
evr_cumsum = np.cumsum(evr)

a = len(np.where(evr_cumsum >= 1.)[0])
b = len(np.where(evr_cumsum >= 0.999)[0])
c = len(np.where(evr_cumsum >= 0.99)[0])
d = len(np.where(evr_cumsum >= 0.95)[0])

num = [a,b,c,d]
acc = []

for p in num :
    pca = PCA(n_components=p)
    pca.fit(x_trn)
    x_trn_P = pca.transform(x_trn)
    x_tst_P = pca.transform(x_tst)
    
    S = time.time()

    #2. 모델
    from catboost import CatBoostRegressor, CatBoostClassifier

    model = CatBoostClassifier(verbose=0)
    
    #3. 컴파일 훈련
    model.fit(x_trn_P, y_trn)
    
    score = model.score(x_tst_P, y_tst)
    
    acc.append((p, score))

for p, ACC in acc:
    print(f"n_components={p:>3} | acc={float(ACC):.4f}")

# n_components=  1 | acc=0.5455
# n_components=  3 | acc=0.6212
# n_components=  4 | acc=0.6364
# n_components=  6 | acc=0.6212    

# CatBoostClassifier R2 : 0.3485
# XGBClassifier R2 : 0.3939
# RandomForestClassifier R2 : 0.4091
# LGBMClassifier R2 : 0.3485
# Stacking score : 0.36363636363636365

# {'target': 0.7121212121212122,
#  'params': {'colsample_bylevel': 0.7716455435126173,
#             'colsample_bytree': 0.8644753639495034,
#             'gamma': 0.08440723019894236,
#             'learning_rate': 0.03370354148846345,
#             'max_bin': 190.04241601832527,
#             'max_depth': 3.3381256602464306,
#             'min_child_samples': 29.86073575276946,
#             'min_child_weight': 5.774438265201903,
#             'n_estimator': 198.16132580542174,
#             'num_leaves': 37.975829671340975,
#             'reg_alpha': 0.5161163930274903,
#             'reg_lambda': 6.048133573318914,
#             'subsample': 0.6193700568007332}}