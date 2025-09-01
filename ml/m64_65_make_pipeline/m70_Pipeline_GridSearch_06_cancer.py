import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from lightgbm import LGBMClassifier

# pipeline : 여러가지 ㅈ전처리를 묶어 놓은 툴!
from sklearn.pipeline import make_pipeline, Pipeline

#########################
#1. 데이터

x, y = load_breast_cancer(return_X_y=True)

x_trn, x_tst, y_trn, y_tst = train_test_split(
     x, y,
     train_size=0.8,
     shuffle=True,
     random_state=777,
     stratify=y
)

parameters = [
{
    "lgbmclassifier__n_estimators": [100, 200, 300],
    "lgbmclassifier__learning_rate": [0.005, 0.05, 0.1],
    "lgbmclassifier__num_leaves": [10, 20, 30]}
]

#2. 모델
pipe = make_pipeline(MinMaxScaler(), LGBMClassifier())

model = GridSearchCV(pipe, parameters, cv = 5, verbose=1, n_jobs=-1)

#3. gnsfus
model.fit(x_trn, y_trn)

#4. 평가 예측
scr = model.score(x_tst, y_tst)
print('scr :', scr)