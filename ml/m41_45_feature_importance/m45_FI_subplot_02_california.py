from sklearn.datasets import fetch_california_housing
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
DS =fetch_california_housing()
x = DS.data
y = DS.target

print(x.shape, y.shape)

x_trn, x_tst, y_trn, y_tst = train_test_split(
    x, y,
    train_size=0.7,
    random_state=seed)

#2 모델구성
model1 = DecisionTreeRegressor(random_state=seed)
model2 = RandomForestRegressor(random_state=seed)
model3 = GradientBoostingRegressor(random_state=seed)
model4 = XGBRFRegressor(random_state=seed)

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

# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ DecisionTreeRegressor ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# acc : 0.6300100448884252
# [0.52040342 0.04301835 0.05031634 0.02555303 0.0322678  0.12844766
#  0.102188   0.0978054 ]
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ RandomForestRegressor ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# acc : 0.8122806484414826
# [0.52718807 0.05144945 0.04904669 0.02972023 0.03227299 0.13212328
#  0.0904717  0.08772758]
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ GradientBoostingRegressor ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# acc : 0.7945394863794644
# [0.60726047 0.02945343 0.02881315 0.00483856 0.00395878 0.1228973
#  0.08630569 0.11647262]
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ XGBRFRegressor ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# acc : 0.7044533857116851
# [0.49104542 0.0641674  0.16197342 0.02030349 0.01370566 0.13761428
#  0.06349537 0.04769505]