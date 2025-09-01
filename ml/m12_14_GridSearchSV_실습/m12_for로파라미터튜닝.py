from sklearn.datasets import load_digits

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

import numpy as np
import random
RS = 44
np.random.seed(RS)
random.seed(RS)
#1. 데이터

x, y = load_digits(return_X_y=True)

x_trn, x_tst, y_trn, y_tst = train_test_split(
    x, y,
    train_size=0.7,
    shuffle=True,
    stratify=y,
    random_state=RS
)

min_samples_split = [2,3,4,5,6] # RandomForestRegressor용
learning_rate = [0.1, 0.05, 0.01, 0.005, 0.001]
max_depth = [3, 4, 5, 6, 7]

best_model = ''
best_score = 0
best_parameters = ''

#2. 모델

for LR in learning_rate:
    for MD in max_depth:
        print('learning_rate :', LR)
        print('max_depth     :', MD)
        model = HistGradientBoostingClassifier(
            learning_rate = LR,
            max_depth = MD,
        )
        
        model.fit(x_trn, y_trn)
        
        score = model.score(x_tst, y_tst)
        
        parameters = f'{LR}, {MD}'
        
        print('점수          :', f"{score:.5f}")
        print('현재최고점수  : {:.5f}'.format(best_score))
        print('현재최적변수  :', best_parameters)
        
        print('~~~~~~~~ 도는중 ~~~~~~~~')
        if score > best_score:
            best_score = score
            best_parameters = parameters


print(best_model)
print(f'최고점수 : {best_score:.5f}')
print('최적변수 :', best_parameters)

# HistGradientBoostingClassifier(learning_rate=0.001, max_depth=7)
# 최고점수 : 0.96852
# 최적변수 : 0.1, 5

# RandomForestRegressor(max_depth=7, min_samples_split=6, verbose=1)
# 최고점수 : 0.79650
# 최적변수 : [6, 7]