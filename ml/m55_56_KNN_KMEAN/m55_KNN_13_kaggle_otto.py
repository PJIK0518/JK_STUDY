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

#####################################
## Scaler
def Scaler(SC, a, b):
    SC.fit(a)
    a_scaled = SC.transform(a)
    b_scaled = SC.transform(b)
    return a_scaled, b_scaled

x_trn, x_tst = Scaler(MaxAbsScaler(), x_trn, x_tst)

#####################################
## 증폭 : class_weight
classes = np.unique(y)
CW = compute_class_weight(class_weight='balanced', classes=classes, y=y)

#####################################
## Label_y
LE = LabelEncoder()
LE.fit(y_trn)
y_trn = LE.transform(y_trn)
y_tst = LE.transform(y_tst)

from xgboost import XGBRegressor, XGBClassifier
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score
import xgboost as xgb

RS = 777

#2 모델구성
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import r2_score, accuracy_score, f1_score
model = KNeighborsClassifier(n_neighbors=5)     # k_neighbors랑 같은 Parameter

model.fit(x_trn, y_trn)
print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ', model.__class__.__name__,'ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print('acc :', model.score(x_tst, y_tst))

y_prd = model.predict(x_tst)

score1 = accuracy_score(y_tst, y_prd)
score2 = f1_score(y_tst, y_prd, average='macro')

print('ACC :',score1)
print('F1S :',score2)

# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ KNeighborsClassifier ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# acc : 0.7608274078862314
# ACC : 0.7608274078862314
# F1S : 0.7018605703114864

# Threshold = 0.003 / n = 93 / R2 = 81.787%

# 최적 컬럼 : 80 개 
#  ['feat_7', 'feat_8', 'feat_9', 'feat_11', 'feat_14', 'feat_15', 
#   'feat_16', 'feat_17', 'feat_18', 'feat_19', 'feat_20', 'feat_21', 
#   'feat_23', 'feat_24', 'feat_25', 'feat_26', 'feat_27', 'feat_29', 
#   'feat_30', 'feat_31', 'feat_32', 'feat_33', 'feat_34', 'feat_35', 
#   'feat_36', 'feat_37', 'feat_38', 'feat_39', 'feat_40', 'feat_41', 
#   'feat_42', 'feat_43', 'feat_44', 'feat_45', 'feat_46', 'feat_47', 
#   'feat_48', 'feat_50', 'feat_51', 'feat_52', 'feat_53', 'feat_54', 
#   'feat_55', 'feat_56', 'feat_57', 'feat_58', 'feat_59', 'feat_60', 
#   'feat_61', 'feat_62', 'feat_63', 'feat_64', 'feat_65', 'feat_66', 
#   'feat_67', 'feat_68', 'feat_69', 'feat_70', 'feat_71', 'feat_72',
#   'feat_73', 'feat_74', 'feat_75', 'feat_76', 'feat_77', 'feat_78', 
#   'feat_79', 'feat_80', 'feat_81', 'feat_83', 'feat_84', 'feat_85',
#   'feat_86', 'feat_87', 'feat_88', 'feat_89', 'feat_90', 'feat_91', 'feat_92', 'feat_93']
# 삭제 컬럼 : 13 개 
#  ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6',
#   'feat_10', 'feat_12', 'feat_13', 'feat_22', 'feat_28', 'feat_49', 'feat_82']
# 최고 점수 81.771%