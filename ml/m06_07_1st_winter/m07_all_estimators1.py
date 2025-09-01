from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
from sklearn.utils import all_estimators

import sklearn as sk
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# print(sk.__version__)   1.6.1

#1. 데이터

x, y = fetch_california_housing(return_X_y=True)

x_trn, x_tst, y_trn, y_tst = train_test_split(
    x, y,
    train_size=0.8,
    shuffle=True,
    random_state=42
)

Scaler = RobustScaler()
Scaler.fit(x_trn)
x_trn = Scaler.transform(x_trn)
x_tst = Scaler.transform(x_tst)

#2. 모델
# model = RandomForestRegressor()

all_Algorithms = all_estimators(type_filter='regressor')

print('all_Algorithms :', all_Algorithms) # sklearn에서 제공하는 regressor 모두 제공
print(len(all_Algorithms))                # 55개

max_score = 0
max_name = 'Default'

for name, model in all_Algorithms:
    try:# 파라미터가 필요한 녀석들 발생! > 파이썬 기초 : 예외 처리
        #3. 훈련
        model = model()
        model.fit(x_trn, y_trn)
        
        #4. 평가 예측
        score = model.score(x_tst, y_tst)
        print(name, '의 정답률 :', score)
        
    except:
        print(name, ': ERROR')
        
    ####### [실습] 위치는 적절히 잘 조절해서...! except 위에도 가능 #############    
    if score > max_score:
        max_score = score
        max_name = name
        print('최고모델 :', max_name, max_score) # 최고모델 : HistGradientBoostingRegressor 0.8350806769001473
    ##########################################################################

''' Tensorflow model... 이거 왜 씀?
0.47339358925819397
[DO]
0.3632981777191162
[CNN]
0.5098786354064941
0.4542386829853058
0.3375522494316101
[LSTM]
0.44303375482559204
[Conv1D]
0.4965818524360657
'''

"""
ARDRegression 의 정답률 : 0.5750762259295374
AdaBoostRegressor 의 정답률 : 0.448426527053309
BaggingRegressor 의 정답률 : 0.7824747763729879
BayesianRidge 의 정답률 : 0.5757963817040466
CCA : ERROR
DecisionTreeRegressor 의 정답률 : 0.6126779086446511
DummyRegressor 의 정답률 : -0.00021908714592466794
ElasticNet 의 정답률 : 0.14040050205675048
ElasticNetCV 의 정답률 : 0.5761196862708703
ExtraTreeRegressor 의 정답률 : 0.549444350609364
ExtraTreesRegressor 의 정답률 : 0.8083121424190013
GammaRegressor 의 정답률 : 0.28691269936460195
GaussianProcessRegressor 의 정답률 : -95.41804749556741
GradientBoostingRegressor 의 정답률 : 0.775640862803052
HistGradientBoostingRegressor 의 정답률 : 0.8356103081593338
HuberRegressor 의 정답률 : 0.561028661989605
IsotonicRegression : ERROR
KNeighborsRegressor 의 정답률 : 0.6757890024425293
KernelRidge 의 정답률 : -1.3489780543322003
Lars 의 정답률 : 0.575787706032451
LarsCV 의 정답률 : 0.575787706032451
Lasso 의 정답률 : -0.00021908714592466794
LassoCV 의 정답률 : 0.5761555625216169
LassoLars 의 정답률 : -0.00021908714592466794
LassoLarsCV 의 정답률 : 0.575787706032451
LassoLarsIC 의 정답률 : 0.575787706032451
LinearRegression 의 정답률 : 0.5757877060324511
LinearSVR 의 정답률 : 0.5319801625206138
MLPRegressor 의 정답률 : 0.7564768456609845
MultiOutputRegressor : ERROR
MultiTaskElasticNet : ERROR
MultiTaskElasticNetCV : ERROR
MultiTaskLasso : ERROR
MultiTaskLassoCV : ERROR
NuSVR 의 정답률 : 0.6662617912081686
OrthogonalMatchingPursuit 의 정답률 : 0.45885918903846656
OrthogonalMatchingPursuitCV 의 정답률 : 0.5097965040568941
PLSCanonical : ERROR
PLSRegression 의 정답률 : 0.5079059853496792
PassiveAggressiveRegressor 의 정답률 : -4.376502517658291
PoissonRegressor 의 정답률 : 0.39526246062883497
QuantileRegressor 의 정답률 : -0.05001307699665003
RANSACRegressor 의 정답률 : 0.3081334104342721
RadiusNeighborsRegressor : ERROR
RandomForestRegressor 의 정답률 : 0.8056063009926673
RegressorChain : ERROR
Ridge 의 정답률 : 0.5758006170065986
RidgeCV 의 정답률 : 0.5757890621564488
SGDRegressor 의 정답률 : -1.4336488798472552e+23
SVR 의 정답률 : 0.6632269374443257
StackingRegressor : ERROR
TheilSenRegressor 의 정답률 : 0.2538800540013457
TransformedTargetRegressor 의 정답률 : 0.5757877060324511
TweedieRegressor 의 정답률 : 0.334531822302523
VotingRegressor : ERROR """


