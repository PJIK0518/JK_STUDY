import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

#1. 데이터

x, y = load_breast_cancer(return_X_y=True)

x_trn, x_tst, y_trn, y_tst = train_test_split(x, y,
                                              random_state=337,
                                              train_size=0.8,
                                              stratify= y)

# print(x.shape)(569, 30)
# print(y.shape)(569,)

parameters = {'n_estimators' : 1000,
              'learning_rate' : 0.3,
              'max_depth' : 3,
              'gamma' : 1,
              'min_child_weight' : 1,
              'subsample' : 1,
              'colsample_bytree' : 1,
              'colsample_bylevel' : 1,
              'colsample_bynode' : 1,
              'reg_alpha' : 0,
              'reg_lambda' : 1,
              'random_state' : 3377,
            #   'verbose' : 0,
              }

#2. 모델
model = XGBClassifier(
    # **parameters            # 외부에 파라미터를 빼두고 불러오는 방식
)                             # 과제!!!! : parameters 앞에 별 하나, 별 두 개의 문법 차이
                              #           본인이 어떻게 이해했는지 정리해서 메일로 보내기


#3. 훈련
model.set_params(**parameters,               # Parameters만 fitting 전에 set_params를 통해서 적용 가능
                 early_stopping_rounds = 10) # early stop을 넣을 수 있음

model.fit(x_trn, y_trn,
          eval_set = [(x_tst, y_tst)],       # early stop 설정하려면 필요
          verbose = 10)

result = model.score(x_tst, y_tst)
print(result)

y_prd = model.predict(x_tst)
acc = accuracy_score(y_tst, y_prd)
print(acc)

path = './_save/m01_job/'
# import joblib
# joblib.dump(model, path + 'm01_joblib_save.joblib') # 확장자는 하고싶은거 아무거나 해도 가능

model.save_model(path + 'm03_xgb_save.dat')
