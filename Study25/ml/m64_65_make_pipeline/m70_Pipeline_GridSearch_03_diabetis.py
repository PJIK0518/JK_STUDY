import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor

# pipeline : 여러가지 ㅈ전처리를 묶어 놓은 툴!
from sklearn.pipeline import Pipeline, make_pipeline

#########################
#1. 데이터

x, y = load_diabetes(return_X_y=True)

x_trn, x_tst, y_trn, y_tst = train_test_split(
     x, y,
     train_size=0.8,
     shuffle=True,
     random_state=777)

parameters = [
    {'xgbregressor__n_estimators': [100, 200],
     'xgbregressor__learning_rate': [0.05, 0.01, 0.005],
     'xgbregressor__min_child_weight': [3, 10]},
    {'xgbregressor__min_child_weight': [6, 8, 10, 12],
     'xgbregressor__subsample': [0.5, 0.7, 0.9]},
    {'xgbregressor__min_child_weight': [3, 5, 7, 9],
     'xgbregressor__reg_alpha': [2, 3, 5, 10]},
    {'xgbregressor__min_child_weight': [2, 3, 5, 6]}
]

#2. 모델
pipe = make_pipeline(MinMaxScaler(), XGBRegressor())


model = GridSearchCV(pipe, parameters, cv = 5, verbose=1, n_jobs=-1)

#3. gnsfus
model.fit(x_trn, y_trn)

#4. 평가 예측
scr = model.score(x_tst, y_tst)
print('scr :', scr)

# scr : 0.3714997258322671