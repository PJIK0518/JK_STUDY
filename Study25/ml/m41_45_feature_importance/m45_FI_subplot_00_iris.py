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

for model in models:
    model.fit(x_trn, y_trn)
    print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ', model.__class__.__name__,'ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
    print('acc :', model.score(x_tst, y_tst))
    print(model.feature_importances_)

def plot_FI_DS(model):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
    axes = axes.flatten()
    feature_names = DS.feature_names
    n_features = x.shape[1]

    for idx, model in enumerate(models):
        axes[idx].barh(range(n_features), model.feature_importances_, align='center')
        axes[idx].set_yticks(range(n_features))
        axes[idx].set_yticklabels(feature_names)
        axes[idx].set_xlabel("Feature Importance")
        axes[idx].set_title(model.__class__.__name__)
        axes[idx].invert_yaxis()

plot_FI_DS(model)

plt.tight_layout()
plt.show()
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ DecisionTreeClassifier ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# acc : 0.8666666666666667
# [0.04199134 0.03428571 0.91033009 0.01339286]
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ RandomForestClassifier ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# acc : 0.9555555555555556
# [0.09264409 0.02733863 0.48236753 0.39764975]
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ GradientBoostingClassifier ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# acc : 0.9555555555555556
# [0.00573464 0.02499731 0.80130425 0.1679638 ]
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ XGBClassifier ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# acc : 0.9555555555555556
# [0.03295855 0.02776272 0.75007254 0.18920612]