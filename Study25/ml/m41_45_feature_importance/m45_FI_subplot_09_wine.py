from sklearn.datasets import load_wine
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
DS =load_wine()
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
# acc : 0.8888888888888888
# [0.         0.         0.         0.         0.         0.
#  0.41323942 0.         0.         0.39919222 0.         0.02396918
#  0.16359918]
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ RandomForestClassifier ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# acc : 0.9814814814814815
# [0.11000244 0.02631156 0.01747922 0.01594035 0.03018239 0.07450764
#  0.16886904 0.01800851 0.02797591 0.14062041 0.07341899 0.12447913
#  0.17220439]
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ GradientBoostingClassifier ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# acc : 0.8888888888888888
# [0.01170521 0.0378643  0.00563892 0.00371273 0.00100388 0.00125986
#  0.3155611  0.00049449 0.00289519 0.31656221 0.00638138 0.03227161
#  0.26464912]
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ XGBClassifier ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# acc : 0.9444444444444444
# [0.0267307  0.06234321 0.02900624 0.01433036 0.00782415 0.0280161
#  0.24978818 0.         0.00537951 0.30230466 0.02314618 0.03196534
#  0.21916535]