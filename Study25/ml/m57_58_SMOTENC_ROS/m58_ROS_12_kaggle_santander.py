import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
import xgboost as xgb
import time
S = time.time()
# 1. 데이터
path = './Study25/_data/kaggle/santander/'

trn_csv = pd.read_csv(path + 'train.csv', index_col=0)
tst_csv = pd.read_csv(path + 'test.csv', index_col=0)
sub_csv = pd.read_csv(path + 'sample_submission.csv')

x = trn_csv.drop(['target'], axis=1)
y = trn_csv['target']

x_trn, x_tst, y_trn, y_tst = train_test_split(
    x, y,
    train_size=0.9,
    shuffle=True,
    random_state=777,
    stratify=y
)
from imblearn.over_sampling import RandomOverSampler

ROS = RandomOverSampler(random_state=337,
                        sampling_strategy='auto')

x_trn, y_trn = ROS.fit_resample(x_trn, y_trn)

# ROS
# F1S : 0.49313304721030043
# ACC : 88.19 %
# 12.0 초

# 기존 : 91.250%
# AUTO : 0.86355 / 12.6초
# JH_SMOTE : 0.87085 | 106.9 초

# 2. 스케일링
def Scaler(SC, a, b):
    SC.fit(a)
    a_scaled = SC.transform(a)
    b_scaled = SC.transform(b)
    return a_scaled, b_scaled

x_trn, x_tst = Scaler(RobustScaler(), x_trn, x_tst)
x_trn, x_tst = Scaler(StandardScaler(), x_trn, x_tst)

# 3. 모델 정의 및 학습
RS = 777

model = XGBClassifier(
    objective='binary:logistic',
    n_estimators=200,
    max_depth=6,
    gamma=0,
    min_child_weight=0,
    subsample=0.4,
    reg_alpha=0,
    reg_lambda=1,
    early_stopping_rounds=10,
    random_state=RS
)

model.fit(x_trn, y_trn,
          eval_set=[(x_tst, y_tst)],
          verbose=0)

y_prd = model.predict(x_tst)

from sklearn.metrics import f1_score
print('F1S :', f1_score(y_prd,y_tst))
print('ACC :', model.score(x_tst,y_tst)*100,'%')
print(f'{(time.time() - S):.1f}',"초")

# 최적 컬럼 : 117 개
#  ['var_0' 'var_1' 'var_2' 'var_5' 'var_6' 'var_9' 'var_12' 'var_13'
#  'var_18' 'var_20' 'var_21' 'var_22' 'var_23' 'var_24' 'var_26' 'var_28'
#  'var_32' 'var_33' 'var_34' 'var_35' 'var_36' 'var_40' 'var_43' 'var_44'
#  'var_48' 'var_49' 'var_51' 'var_53' 'var_55' 'var_56' 'var_57' 'var_58'
#  'var_66' 'var_67' 'var_70' 'var_71' 'var_75' 'var_76' 'var_78' 'var_80'
#  'var_81' 'var_82' 'var_86' 'var_87' 'var_88' 'var_89' 'var_91' 'var_92'
#  'var_93' 'var_94' 'var_95' 'var_99' 'var_104' 'var_105' 'var_106'
#  'var_107' 'var_108' 'var_109' 'var_110' 'var_111' 'var_112' 'var_114'
#  'var_115' 'var_117' 'var_118' 'var_119' 'var_121' 'var_122' 'var_123'
#  'var_125' 'var_127' 'var_128' 'var_130' 'var_131' 'var_132' 'var_133'
#  'var_135' 'var_137' 'var_138' 'var_139' 'var_141' 'var_145' 'var_146'
#  'var_147' 'var_148' 'var_149' 'var_150' 'var_151' 'var_154' 'var_155'
#  'var_157' 'var_162' 'var_163' 'var_164' 'var_165' 'var_166' 'var_167'
#  'var_169' 'var_170' 'var_172' 'var_173' 'var_174' 'var_175' 'var_177'
#  'var_179' 'var_180' 'var_184' 'var_186' 'var_187' 'var_188' 'var_190'
#  'var_191' 'var_192' 'var_194' 'var_195' 'var_197' 'var_198']
# 삭제 컬럼 : 83 개
#  ['var_3' 'var_4' 'var_7' 'var_8' 'var_10' 'var_11' 'var_14' 'var_15'
#  'var_16' 'var_17' 'var_19' 'var_25' 'var_27' 'var_29' 'var_30' 'var_31'
#  'var_37' 'var_38' 'var_39' 'var_41' 'var_42' 'var_45' 'var_46' 'var_47'
#  'var_50' 'var_52' 'var_54' 'var_59' 'var_60' 'var_61' 'var_62' 'var_63'
#  'var_64' 'var_65' 'var_68' 'var_69' 'var_72' 'var_73' 'var_74' 'var_77'
#  'var_79' 'var_83' 'var_84' 'var_85' 'var_90' 'var_96' 'var_97' 'var_98'
#  'var_100' 'var_101' 'var_102' 'var_103' 'var_113' 'var_116' 'var_120'
#  'var_124' 'var_126' 'var_129' 'var_134' 'var_136' 'var_140' 'var_142'
#  'var_143' 'var_144' 'var_152' 'var_153' 'var_156' 'var_158' 'var_159'
#  'var_160' 'var_161' 'var_168' 'var_171' 'var_176' 'var_178' 'var_181'
#  'var_182' 'var_183' 'var_185' 'var_189' 'var_193' 'var_196' 'var_199']
# 최고 점수 : 91.250%