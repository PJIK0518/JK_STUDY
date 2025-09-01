# Votting : 기본적으로 2개 이상, 이상적으로 3개 이상의 모델을 통해 가장 적절한 결과값 산출
          # Hard_voting > 각 모델에서 나온 결과값을 기준으로 많이 나온 결과를 최종 산출
          # Soft_voting > 각 모델에서 나온 확률값을 합산하여 높게 나온 결과를 최종 산출

import warnings

warnings.filterwarnings('ignore')


import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingClassifier

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

from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

x_trn, x_tst, y_trn, y_tst = train_test_split(
    x, y,
    train_size=0.9,
    shuffle=True,
    random_state=777,
    stratify=y
)

RS = 42

from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score
import xgboost as xgb

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

from xgboost.plotting import plot_importance
import matplotlib.pyplot as plt
plot_importance(model)

print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ', model.__class__.__name__,'ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print('R2 :', model.score(x_tst, y_tst))
print(model.feature_importances_)

# weight = model.get_booster().get_score(importance_type = 'weight') # Frequecny
gain = model.get_booster().get_score(importance_type = 'gain')     # DataFrame 형태로 저장

""" [해설_Teacher] """
total = sum(gain.values())

print(total)

                            # Nan 값이 있으면 0으로 채워라
gain_list = [gain.get(f'f{i}', 0) / total for i in range(x.shape[1])]  # Dictionary.get('Key') > Dictionary Key에 대한 Value 값 송출
                                                                       # 첫 번째 f : formatted string으로 줄여서 f-string
                                                                       # 두 번째 f : 그냥 문자 f
                                                                       # i : 0~29 순서대로 입력 > gain.get(f'f0') ~ gain.get(f'f29')
# print(gain_list)
# print(len(gain_list))

thresholds = np.sort(gain_list)
# print(thresholds)
# [1.39180242e-04 1.04901660e-03 3.38101714e-03 3.76765075e-03
#  4.28204740e-03 4.52440786e-03 5.80562371e-03 6.88414928e-03
#  8.52820792e-03 9.58394778e-03 1.23838499e-02 1.36080448e-02
#  1.37176080e-02 1.42034000e-02 1.72352618e-02 1.77809649e-02
#  1.83013105e-02 2.12252283e-02 2.12534293e-02 2.38961533e-02
#  2.61276730e-02 2.78357997e-02 2.90817329e-02 3.14597693e-02
#  4.53606655e-02 5.57694243e-02 1.00636591e-01 1.07826644e-01
#  1.76796239e-01 1.77554961e-01]

""" [실습_JK]
GAIN = []

for i in gain:
    G = gain[i]
    GAIN.append(G)

min = min(GAIN)
max = max(GAIN)

GAIN = [(g-min)/(max -min) for g in GAIN]

thresholds = np.sort(GAIN) """

feature_names = DS.feature_names

""" print(feature_names)
['mean radius' 'mean texture' 'mean perimeter' 'mean area'
 'mean smoothness' 'mean compactness' 'mean concavity'
 'mean concave points' 'mean symmetry' 'mean fractal dimension'
 'radius error' 'texture error' 'perimeter error' 'area error'
 'smoothness error' 'compactness error' 'concavity error'
 'concave points error' 'symmetry error' 'fractal dimension error'
 'worst radius' 'worst texture' 'worst perimeter' 'worst area'
 'worst smoothness' 'worst compactness' 'worst concavity'
 'worst concave points' 'worst symmetry' 'worst fractal dimension'] """
# ### 컬럼명 매칭 ###
# score_DF = pd.DataFrame({
#     'feature' : feature_names,   # gain의 Key : f0~f29 > f이후의 값을 int화 > feature_names의 int 번째 값을 입력
#     'gain' : gain_list                                   # gain의 value 값을 순서대로 입력
# }).sort_values(by='gain', ascending=True)

# Dataframe 활용 언어 : DF.keys()
                    #   DF.valuse()
                    #   DF.sort_values(by='gain', ascending='True'or'Fasle')
# print(score_DF)
# exit()
from sklearn.feature_selection import SelectFromModel

BEST_drp = []
BEST_scr = 0
BEST_trn = x_trn

for i in thresholds:
    selection = SelectFromModel(model,
                                threshold=i,
                                prefit=False)
     
    select_x_trn = selection.transform(x_trn)
    select_x_tst = selection.transform(x_tst)
    
    select_model = XGBClassifier(
    n_estimators = 200,
    max_depth = 6,
    gamma = 0,
    min_child_weight = 0,
    subsample = 0.4,
    reg_alpha = 0,
    reg_lambda = 1,                 # CHAT GPT에 각각 계산 방식 물어보기 #
    eval_metric = 'logloss',        # 다중 분류 : mlogloss, merror
                                    # 이진 분류 : logloss, error
                                    # 회귀      : rmse, mae, rmsle
                                    # 2.1.1 버전 이후로 fit 에서 모델로 위치 변경
    early_stopping_rounds=10,
    random_state=RS)
    
    select_model.fit(select_x_trn, y_trn,
                     eval_set = [(select_x_tst,y_tst)],
                     verbose = False)
                
    score = select_model.score(select_x_tst,y_tst)

    Columns = selection.get_support()       # SelectFromModel에서 지정한 threshold를 기준으로 

    Drop = [feature_names[j]
            for j, selected in enumerate(Columns)
            if not selected] # 선생님 스타일
    
    Droped = [not i for i in Columns]
    C_feature = feature_names[Columns]
    D_feature = feature_names[Droped]
    
    if BEST_scr <= score:
       BEST_scr = score
       BEST_trn = select_x_trn
       BEST_col = C_feature
       BEST_drp = D_feature
        
    print(f'Threshold = {i:.3f} / n = {select_x_trn.shape[1]:2d} / R2 = {score*100:.3f}%')
    print(C_feature)
    print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')

print('최적 컬럼 :', BEST_trn.shape[1],'개','\n',
      BEST_col)
print('삭제 컬럼 :',f'{x_trn.shape[1]-BEST_trn.shape[1]}','개','\n',
      BEST_drp)
print('최고 점수', f'{BEST_scr*100:.3f}%')

""" Threshold = 0.000 / n = 30 / R2 = 100.000%
['mean radius' 'mean texture' 'mean perimeter' 'mean area'
 'mean smoothness' 'mean compactness' 'mean concavity'
 'mean concave points' 'mean symmetry' 'mean fractal dimension'
 'radius error' 'texture error' 'perimeter error' 'area error'
 'smoothness error' 'compactness error' 'concavity error'
 'concave points error' 'symmetry error' 'fractal dimension error'
 'worst radius' 'worst texture' 'worst perimeter' 'worst area'
 'worst smoothness' 'worst compactness' 'worst concavity'
 'worst concave points' 'worst symmetry' 'worst fractal dimension']
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
Threshold = 0.005 / n = 24 / R2 = 98.246%
['mean texture' 'mean perimeter' 'mean area' 'mean compactness'
 'mean concavity' 'mean concave points' 'mean symmetry'
 'mean fractal dimension' 'perimeter error' 'area error'
 'smoothness error' 'compactness error' 'concavity error'
 'concave points error' 'symmetry error' 'fractal dimension error'
 'worst radius' 'worst texture' 'worst perimeter' 'worst area'
 'worst smoothness' 'worst concavity' 'worst concave points'
 'worst symmetry']
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
Threshold = 0.018 / n = 14 / R2 = 98.246%
['mean area' 'mean concave points' 'perimeter error' 'area error'
 'concave points error' 'symmetry error' 'fractal dimension error'
 'worst radius' 'worst texture' 'worst perimeter' 'worst area'
 'worst smoothness' 'worst concavity' 'worst concave points']
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
Threshold = 0.020 / n = 13 / R2 = 100.000%
['mean area' 'mean concave points' 'perimeter error' 'area error'
 'concave points error' 'symmetry error' 'fractal dimension error'
 'worst radius' 'worst texture' 'worst perimeter' 'worst area'
 'worst concavity' 'worst concave points']
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
Threshold = 0.023 / n = 11 / R2 = 98.246%
['mean area' 'mean concave points' 'perimeter error' 'area error'
 'symmetry error' 'worst radius' 'worst texture' 'worst perimeter'
 'worst area' 'worst concavity' 'worst concave points']
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
Threshold = 0.025 / n = 10 / R2 = 98.246%
['mean area' 'mean concave points' 'perimeter error' 'area error'
 'worst radius' 'worst texture' 'worst perimeter' 'worst area'
 'worst concavity' 'worst concave points']
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
Threshold = 0.032 / n =  6 / R2 = 92.982%
['mean area' 'mean concave points' 'worst perimeter' 'worst area'
 'worst concavity' 'worst concave points']
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
Threshold = 0.038 / n =  6 / R2 = 92.982%
['mean area' 'mean concave points' 'worst perimeter' 'worst area'
 'worst concavity' 'worst concave points']
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
Threshold = 0.047 / n =  5 / R2 = 94.737%
['mean area' 'mean concave points' 'worst perimeter' 'worst area'
 'worst concave points']
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
Threshold = 0.053 / n =  5 / R2 = 94.737%
['mean area' 'mean concave points' 'worst perimeter' 'worst area'
 'worst concave points']
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
Threshold = 0.069 / n =  4 / R2 = 94.737%
['mean area' 'worst perimeter' 'worst area' 'worst concave points']
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
Threshold = 0.076 / n =  4 / R2 = 94.737%
['mean area' 'worst perimeter' 'worst area' 'worst concave points']
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
Threshold = 0.077 / n =  4 / R2 = 94.737%
['mean area' 'worst perimeter' 'worst area' 'worst concave points']
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
Threshold = 0.079 / n =  4 / R2 = 94.737%
['mean area' 'worst perimeter' 'worst area' 'worst concave points']
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
Threshold = 0.096 / n =  4 / R2 = 94.737%
['mean area' 'worst perimeter' 'worst area' 'worst concave points']
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
Threshold = 0.099 / n =  4 / R2 = 94.737%
['mean area' 'worst perimeter' 'worst area' 'worst concave points']
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
Threshold = 0.102 / n =  3 / R2 = 94.737%
['mean area' 'worst perimeter' 'worst concave points']
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
Threshold = 0.119 / n =  2 / R2 = 94.737%
['mean area' 'worst concave points']
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
Threshold = 0.119 / n =  2 / R2 = 94.737%
['mean area' 'worst concave points']
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
Threshold = 0.134 / n =  2 / R2 = 94.737%
['mean area' 'worst concave points']
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
Threshold = 0.146 / n =  2 / R2 = 94.737%
['mean area' 'worst concave points']
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
Threshold = 0.156 / n =  2 / R2 = 94.737%
['mean area' 'worst concave points']
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
Threshold = 0.163 / n =  2 / R2 = 94.737%
['mean area' 'worst concave points']
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
Threshold = 0.177 / n =  2 / R2 = 94.737%
['mean area' 'worst concave points']
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
Threshold = 0.255 / n =  0 /      SKIP
Threshold = 0.314 / n =  0 /      SKIP
Threshold = 0.566 / n =  0 /      SKIP
Threshold = 0.607 / n =  0 /      SKIP
Threshold = 0.996 / n =  0 /      SKIP
Threshold = 1.000 / n =  0 /      SKIP """