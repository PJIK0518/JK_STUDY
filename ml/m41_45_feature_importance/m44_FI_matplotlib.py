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

def plot_FI_DS(model):
    n_features = DS.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), model.feature_importances_)
    plt.xlabel('feature_importances')
    plt.ylabel('feature')
    plt.ylim(-1, n_features)
    plt.title(model.__class__.__name__)
    
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

for model in models:
    model.fit(x_trn, y_trn)
    print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ', model.__class__.__name__,'ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
    print('acc :', model.score(x_tst, y_tst))
    print(model.feature_importances_)

plot_FI_DS(model)
plt.show()