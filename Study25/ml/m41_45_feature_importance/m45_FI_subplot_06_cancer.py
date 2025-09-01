from sklearn.datasets import load_breast_cancer
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
DS =load_breast_cancer()
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
# acc : 0.9239766081871345
# [0.00000000e+00 6.56307339e-02 1.22934363e-02 2.00774796e-03
#  0.00000000e+00 1.52387387e-02 0.00000000e+00 0.00000000e+00
#  0.00000000e+00 7.17117117e-03 2.45565616e-03 0.00000000e+00
#  0.00000000e+00 0.00000000e+00 0.00000000e+00 8.60540541e-03
#  0.00000000e+00 5.37837838e-03 0.00000000e+00 0.00000000e+00
#  0.00000000e+00 5.83407477e-04 6.80776868e-01 4.72193596e-02
#  5.05329393e-03 0.00000000e+00 2.34026134e-02 1.16115623e-01
#  8.06756757e-03 0.00000000e+00]
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ RandomForestClassifier ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# acc : 0.9649122807017544
# [0.03751152 0.01750919 0.05148285 0.04084838 0.00500837 0.01346779
#  0.04823759 0.09323339 0.00444285 0.00424179 0.01089097 0.00626193
#  0.01585402 0.04437793 0.00326862 0.00493032 0.00564358 0.00552437
#  0.00418224 0.00547716 0.10026747 0.01622894 0.12137365 0.12357732
#  0.01189728 0.01117922 0.02921973 0.14527489 0.01308676 0.00549986]
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ GradientBoostingClassifier ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# acc : 0.9941520467836257
# [1.25918107e-03 3.20456049e-02 5.55441258e-03 4.15639661e-04
#  2.55034194e-04 3.97070833e-03 1.06239056e-02 3.86084764e-02
#  4.03506288e-04 3.52200832e-04 2.92403641e-03 4.81008031e-03
#  4.48396300e-05 5.05328138e-03 1.82652809e-03 7.88296397e-05
#  2.25870016e-03 1.55173739e-03 1.11442576e-04 3.36640166e-03
#  1.27039653e-02 3.16066865e-02 3.72742222e-01 3.19931131e-01
#  1.08092273e-02 7.30192503e-04 2.45169078e-02 1.10283655e-01
#  1.11572226e-03 4.57430909e-05]
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ XGBClassifier ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# acc : 0.9883040935672515
# [0.01964068 0.03524631 0.02581123 0.02879266 0.00380721 0.00603373
#  0.01990849 0.05847218 0.00143627 0.00920418 0.00749438 0.
#  0.00598944 0.00659488 0.00337986 0.00431487 0.00122592 0.01863656
#  0.00303041 0.00472867 0.04450388 0.01634623 0.48271778 0.02247635
#  0.01861823 0.         0.02894365 0.11843071 0.00421524 0.        ]