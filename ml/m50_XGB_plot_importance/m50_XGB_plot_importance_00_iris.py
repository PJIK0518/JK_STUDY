from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRFRegressor
import random
import numpy as np

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

seed = 123
random.seed(seed)
np.random.seed(seed)

    
#1 데이터
DS =load_iris()
x = DS.data
y = DS.target

print(x.shape, y.shape)

x_trn, x_tst, y_trn, y_tst = train_test_split(
    x, y,
    train_size=0.7,
    random_state=seed,
    stratify=y)

#2 모델구성
model1 = DecisionTreeClassifier(random_state=seed)
model2 = RandomForestClassifier(random_state=seed)
model3 = GradientBoostingClassifier(random_state=seed)
model4 = XGBClassifier(random_state=seed)

models = [model1,model2,model3,model4]
model4.fit(x_trn, y_trn)

from xgboost.plotting import plot_importance
plot_importance(model4,
                importance_type = 'gain',
                title = 'FI [GAIN]')
plt.show()

# 각 Feature의 Frequency를 기준으로 중요도표시
# importance_type = 'weight' : Frequency
# 트리 구조에서 결과값을 찾아내기위해 spliting 빈도수(트리 구조에서 가지 치기 된 횟수)

# importance_type = 'gain'
# spliting이 모델 성능 개선에 얼마나 기여했는가 : 통상적으로 사용되는 수치

# importance_type = 'cover'
# spliting하기 위한 Sample 수 별로