# SMOTE의 문제점은??
# 보간법 : 실제 데이터가 1 과 10이면 1~10사이의 값에서 데이터 증폭
# >> 범주형 데이터라면?? 0 1으로 나와야하는데 0.5가 나오면??

# m10_13.copy

# tst, val : 데이터가 아깝다...
# But. 데이터의 과적합을 막기 위해서 필요한 과정
#      But. tst, val에 중요도가 높은 애들이 있다면??
#           >> tst에 한 번 썼던 데이터를 trn에 넣고 기존 trn에서 다시 tst 선정
#           >> 데이터 1/n 로 나눠서 n 반복 : n_split (데이터가 많다면 n 수를 높여서 진행)
#           >> 데이터의 소실 없이 훈련 가능7
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
from sklearn.preprocessing import MaxAbsScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import time
S= time.time()
#1. 데이터
path = './Study25/_data/kaggle/otto/'

trn_csv = pd.read_csv(path + 'train.csv', index_col=0)
tst_csv = pd.read_csv(path + 'test.csv', index_col=0)
sub_csv = pd.read_csv(path + 'sampleSubmission.csv')

x = trn_csv.drop(['target'], axis=1)
y = trn_csv['target']

x_trn, x_tst, y_trn, y_tst = train_test_split(
    x, y,
    train_size=0.9,
    shuffle=True,
    random_state=777
)
# print(trn_csv.info())
# exit()

from imblearn.over_sampling import SMOTE
print(np.unique(y, return_counts=True))
# (array(['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6',
    #    'Class_7', 'Class_8', 'Class_9'], dtype=object), array([ 1929, 16122,  8004,  2691,  2739, 14135,  2839,  8464,  4955]))
# exit()
#####################################
## Label_y
LE = LabelEncoder()
LE.fit(y_trn)
y_trn = LE.transform(y_trn)
y_tst = LE.transform(y_tst)

# SMT = SMOTE(random_state=42,
#             # sampling_strategy='auto',
#             sampling_strategy={0:20000,
#                                 1:20000,
#                                 2:20000,
#                                 3:20000,
#                                 4:20000,
#                                 5:20000,
#                                 6:20000,
#                                 7:20000,
#                                 8:20000,
#                           },
#             )

# x_trn, y_trn = SMT.fit_resample(x_trn, y_trn)

# 기존 : 최고 점수 81.771%

# AUTO
# F1S : 0.7713276942715936
# ACC : 80.18745959922431 %
# 8.9 초

# JH_SMOTE
# F1S : 0.77591390699954
# ACC : 80.62378797672916 %
# 12.3 초

y_trn = y_trn.astype(int)
from imblearn.over_sampling import SMOTENC ## 범주형 데이터 증폭용!

# 데이터 20등분하기
n_splits = 20
split_size = len(x_trn) // n_splits

x_augmented = []
y_augmented = []

categorical_features = list(range(90))

for i in range(n_splits):
    start = i * split_size
    end = (i + 1) * split_size if i != n_splits - 1 else len(x_trn)
    
    x_batch = x_trn[start:end]
    y_batch = y_trn[start:end]
    
    sampling_strategy = {cls: 20000 for cls in np.unique(y_batch)}
    
    SMTNC = SMOTENC(
        random_state=337,
        categorical_features=categorical_features,
        sampling_strategy= sampling_strategy,
    )
    
    x_res, y_res = SMTNC.fit_resample(x_batch, y_batch)

    
    x_augmented.append(x_res)
    y_augmented.append(y_res)
# 20개 분할 결과 합친 후:
x_trn = np.vstack(x_augmented)
y_trn = np.hstack(y_augmented)

# DataFrame/Series로 변환하여 feature names 보존
x_trn = pd.DataFrame(x_trn, columns=x.columns)
y_trn = pd.Series(y_trn)
# 최고 점수 86.312%
# SMOTE    : 최고 점수 86.119%
# JH_SMOTE : 최고 점수 85.925%

from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score
import xgboost as xgb
RS = 77
#2 모델구성
model = XGBClassifier(
    n_estimators = 200,
    max_depth = 6,
    gamma = 0,
    min_child_weight = 0,
    subsample = 0.4,
    reg_alpha = 0,
    reg_lambda = 1,                 # CHAT GPT에 각각 계산 방식 물어보기 #
 #  eval_metric = 'mlogloss',       # 다중 분류 : mlogloss, merror
                                    # 이진 분류 : logloss, error
                                    # 2.1.1 버전 이후로 fit 에서 모델로 위치 변경
    early_stopping_rounds=10,
    random_state=RS
    )

model.fit(x_trn, y_trn,
          eval_set = [(x_tst,y_tst)],
          verbose = 0)

gain = model.get_booster().get_score(importance_type = 'gain')

total = sum(gain.values())

# print(total)

                            # Nan 값이 있으면 0으로 채워라
gain_list = [i / total for i in gain.values()] 
# print(gain_list)

# print(len(gain_list))



thresholds = np.sort(gain_list)

feature_names = np.array(x.columns)

from sklearn.feature_selection import SelectFromModel

BEST_col = []
BEST_drp = []
BEST_scr = 0
BEST_trn = x_trn

for i in thresholds:
    selection = SelectFromModel(model,
                                threshold=i,
                                prefit=False)
    
    select_x_trn_s = selection.transform(x_trn)
    select_x_tst_s = selection.transform(x_tst)
    
    Columns = selection.get_support()       
    selected_features = feature_names[Columns]
    
    select_x_trn_S = pd.DataFrame(select_x_trn_s, columns=selected_features)
    select_x_tst_S = pd.DataFrame(select_x_tst_s, columns=selected_features)  # <-- 올바르게 수정
    
    select_model = XGBClassifier(
                    n_estimators = 200,
                    max_depth = 6,
                    gamma = 0,
                    min_child_weight = 0,
                    subsample = 0.4,
                    reg_alpha = 0,
                    reg_lambda = 1,                 
                    eval_metric = 'logloss',
                    early_stopping_rounds=10,
                    random_state=42)
    
    if select_x_trn_S.shape[1] == 0:
        print('DONE')
        continue
    
    select_model.fit(select_x_trn_S, y_trn,
                     eval_set = [(select_x_tst_S,y_tst)],
                     verbose = False)
                
    score = select_model.score(select_x_tst_S,y_tst)

    Droped = [not i for i in Columns]
    C_feature = feature_names[Columns]
    D_feature = feature_names[Droped]
    
    if BEST_scr <= score:
        BEST_scr = score
        BEST_trn = select_x_trn_S
        BEST_col = list(C_feature)
        BEST_drp = list(D_feature)
        
    print(f'Threshold = {i:.3f} / n = {select_x_trn_S.shape[1]:2d} / R2 = {score*100:.3f}%')
    print(C_feature)
    print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')

print('최적 컬럼 :', BEST_trn.shape[1],'개','\n',
      BEST_col)
print('삭제 컬럼 :',f'{x_trn.shape[1]-BEST_trn.shape[1]}','개','\n',
      BEST_drp)
print('최고 점수', f'{BEST_scr*100:.3f}%')

# 최적 컬럼 : 6 개 
#  ['Geography', 'Gender', 'Age', 'Balance', 'NumOfProducts', 'IsActiveMember']
# 삭제 컬럼 : 4 개 
#  ['CreditScore', 'Tenure', 'HasCrCard', 'EstimatedSalary']
# 최고 점수 86.555%