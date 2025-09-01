import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
import xgboost as xgb

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

# 2. 스케일링
def Scaler(SC, a, b):
    SC.fit(a)
    a_scaled = SC.transform(a)
    b_scaled = SC.transform(b)
    return a_scaled, b_scaled

x_trn, x_tst = Scaler(RobustScaler(), x_trn, x_tst)
x_trn, x_tst = Scaler(StandardScaler(), x_trn, x_tst)

# print(np.unique(y, return_counts=True))

# exit()
#2 모델구성
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, accuracy_score, f1_score

nc = 2
model1 = KMeans(n_clusters=nc,      
               init='k-means++',   
               n_init=50,

               )
model2 = KMeans(n_clusters=nc,     
               init='k-means++',
               n_init=50,         
               random_state=34,
               )
model3 = KMeans(n_clusters=nc,       
               init='k-means++',   
               n_init=50,         
               random_state=56,
               )
model4 = KMeans(n_clusters=nc,       
               init='k-means++',   
               n_init=50,         
               random_state=78,
               )
model5 = KMeans(n_clusters=nc,       
               init='k-means++',   
               n_init=50,         
               random_state=90,
               )

y_trn_prd1 = model1.fit_predict(x_trn)
y_tst_prd1 = model1.predict(x_tst)

y_trn_prd2 = model2.fit_predict(x_trn)
y_tst_prd2 = model2.predict(x_tst)

y_trn_prd3 = model3.fit_predict(x_trn)
y_tst_prd3 = model3.predict(x_tst)

y_trn_prd4 = model4.fit_predict(x_trn)
y_tst_prd4 = model4.predict(x_tst)

y_trn_prd5 = model5.fit_predict(x_trn)
y_tst_prd5 = model5.predict(x_tst)

y_votes = np.vstack([y_tst_prd1, y_tst_prd2, y_tst_prd3, y_tst_prd4, y_tst_prd5])

from scipy.stats import mode
y_vote_result, _ = mode(y_votes, axis=0, keepdims=False)
print(y_tst_prd1[0])
print(y_tst_prd2[0])
print(y_tst_prd3[0])
print(y_tst_prd4[0])
print(y_tst_prd5[0])
print(y_votes[0])
print(y_vote_result[0])
exit()
score1 = accuracy_score(y_tst, y_vote_result)
score2 = f1_score(y_tst, y_vote_result, average='macro')

print('✅ Manual Hard Voting for KMeans ensemble')
print('ACC :', score1)
print('F1S :', score2)

# ACC : 0.61665
# F1S : 0.5244338633098309

# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ KNeighborsClassifier ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# acc : 0.89945
# ACC : 0.89945
# F1S : 0.4750149882437686

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