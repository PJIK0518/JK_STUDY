from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRFRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBClassifier
import random
import numpy as np

from sklearn.model_selection import train_test_split

seed = 25
random.seed(seed)
np.random.seed(seed)

#1. 데이터
DS1 = load_diabetes()
DS2 = fetch_california_housing()

DSL = [DS1, DS2]
DS_name = ['diabetes', 'california']

model1 = DecisionTreeRegressor(random_state=seed)
model2 = RandomForestRegressor(random_state=seed)
model3 = GradientBoostingRegressor(random_state=seed)
model4 = XGBRFRegressor(random_state=seed)
models = [model1,model2,model3,model4]

for i, DS in enumerate(DSL):
    x = DS.data
    y = DS.target
    
    x_trn, x_tst, y_trn, y_tst = train_test_split(
        x, y,
        train_size=0.85,
        shuffle=True,
        random_state=777
    )
    
    SCL = StandardScaler()
    x_trn = SCL.fit_transform(x_trn)
    x_tst = SCL.transform(x_tst)
    
    print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ',DS_name[i],'ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
    for model in models:
        model.fit(x_trn, y_trn)
        print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ', model.__class__.__name__,'ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
        print('acc :', model.score(x_tst, y_tst))
        print(model.feature_importances_)

# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ diabetes ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ DecisionTreeRegressor ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# acc : 0.022469019337081564
# [0.05165903 0.00510407 0.26468146 0.06147738 0.07370808 0.05616463
#  0.0375994  0.01231021 0.36645509 0.07084063]
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ RandomForestRegressor ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# acc : 0.3469820835375531
# [0.06654054 0.00908053 0.27330628 0.10093072 0.0505745  0.05737914
#  0.05335223 0.02424867 0.30022047 0.06436691]
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ GradientBoostingRegressor ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# acc : 0.39528544105377594
# [0.0562449  0.00873038 0.252909   0.10503027 0.02986105 0.05275674
#  0.03984248 0.01520801 0.38177715 0.05764001]
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ XGBRFRegressor ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# acc : 0.40044574081354
# [0.02552837 0.02461571 0.19444145 0.08261846 0.04719694 0.05791425
#  0.0542576  0.08163789 0.35339987 0.07838945]
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ california ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ DecisionTreeRegressor ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# acc : 0.5752262562356231
# [0.52101023 0.05186871 0.05350251 0.02712911 0.03075654 0.13602692
#  0.09568536 0.08402062]
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ RandomForestRegressor ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# acc : 0.8119907012762907
# [0.52217645 0.05081712 0.04290499 0.02958786 0.03111596 0.13675865
#  0.09440786 0.09223111]
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ GradientBoostingRegressor ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# acc : 0.7865117063127011
# [0.60019605 0.03050371 0.02154205 0.00498166 0.00293472 0.12889976
#  0.09618287 0.11475919]
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ XGBRFRegressor ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# acc : 0.6947050505042603
# [0.46768084 0.05322575 0.19353306 0.02027472 0.01275656 0.14883937
#  0.05967398 0.04401574]