from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRFRegressor
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
import random
import numpy as np

from sklearn.model_selection import train_test_split

seed = 25
random.seed(seed)
np.random.seed(seed)

#1. 데이터
DS = load_diabetes()

x = DS.data
y = DS.target

x_trn, x_tst, y_trn, y_tst = train_test_split(
    x, y,
    train_size=0.85,
    shuffle=True,
    random_state=777
)

###############################
MS = MinMaxScaler()

MS.fit(x_trn)

x_trn = MS.transform(x_trn)
# x_tst = MS.transform(x_tst)

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
    
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ DecisionTreeRegressor ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# acc : -0.09504149716657784
# [0.05136517 0.00665089 0.25950328 0.06551497 0.06738691 0.06068907
#  0.03668295 0.00905705 0.36257414 0.08057557]
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ RandomForestRegressor ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# acc : -0.2728548880455124
# [0.06634741 0.00962675 0.26690581 0.0929402  0.04688972 0.05584583
#  0.05269389 0.02268555 0.31447918 0.07158566]
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ GradientBoostingRegressor ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# acc : 0.01439840231763001
# [0.05657143 0.00858244 0.25330393 0.10332788 0.03018872 0.05059796
#  0.04052034 0.01433091 0.38469409 0.05788229]
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ XGBRFRegressor ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# acc : -0.4060703931364842
# [0.02778012 0.02623826 0.21391177 0.08482479 0.04714009 0.05874474
#  0.06310283 0.08116368 0.32103363 0.07606002]