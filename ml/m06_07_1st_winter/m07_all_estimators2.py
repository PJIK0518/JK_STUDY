#m07_1.copy

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
from sklearn.utils import all_estimators

import sklearn as sk
import numpy as np
import warnings
import pandas as pd
warnings.filterwarnings('ignore')


# print(sk.__version__)   1.6.1

#1. 데이터
path = './_data/kaggle/bank/'

trn_csv = pd.read_csv(path + 'train.csv', index_col=0)
'''print(trn_csv) [165034 rows x 13 columns]
        CustomerId         Surname  CreditScore Geography  Gender   Age  Tenure    Balance  NumOfProducts  HasCrCard  IsActiveMember  EstimatedSalary  Exited
id
0         15674932  Okwudilichukwu          668    France    Male  33.0       3       0.00              2        1.0             0.0        181449.97       0
1         15749177   Okwudiliolisa          627    France    Male  33.0       1       0.00              2        1.0             1.0         49503.50       0
2         15694510           Hsueh          678    France    Male  40.0      10       0.00              2        1.0             0.0        184866.69       0
3         15741417             Kao          581    France    Male  34.0       2  148882.54              1        1.0             1.0         84560.88       0
4         15766172       Chiemenam          716     Spain    Male  33.0       5       0.00              2        1.0             1.0         15068.83       0
...            ...             ...          ...       ...     ...   ...     ...        ...            ...        ...             ...              ...     ...
165029    15667085            Meng          667     Spain  Female  33.0       2       0.00              1        1.0             1.0        131834.75       0
165030    15665521       Okechukwu          792    France    Male  35.0       3       0.00              1        0.0             0.0        131834.45       0
165031    15664752            Hsia          565    France    Male  31.0       5       0.00              1        1.0             1.0        127429.56       0
165032    15689614          Hsiung          554     Spain  Female  30.0       7  161533.00              1        0.0             1.0         71173.03       0
165033    15732798         Ulyanov          850    France    Male  31.0       1       0.00              1        1.0             0.0         61581.79       1
'''
'''print(trn_csv) : index_col = 0 
        CustomerId         Surname  CreditScore Geography  Gender   Age  Tenure    Balance  NumOfProducts  HasCrCard  IsActiveMember  EstimatedSalary  Exited
id
0         15674932  Okwudilichukwu          668    France    Male  33.0       3       0.00              2        1.0             0.0        181449.97       0
1         15749177   Okwudiliolisa          627    France    Male  33.0       1       0.00              2        1.0             1.0         49503.50       0
2         15694510           Hsueh          678    France    Male  40.0      10       0.00              2        1.0             0.0        184866.69       0
3         15741417             Kao          581    France    Male  34.0       2  148882.54              1        1.0             1.0         84560.88       0
4         15766172       Chiemenam          716     Spain    Male  33.0       5       0.00              2        1.0             1.0         15068.83       0
...            ...             ...          ...       ...     ...   ...     ...        ...            ...        ...             ...              ...     ...
165029    15667085            Meng          667     Spain  Female  33.0       2       0.00              1        1.0             1.0        131834.75       0
165030    15665521       Okechukwu          792    France    Male  35.0       3       0.00              1        0.0             0.0        131834.45       0
165031    15664752            Hsia          565    France    Male  31.0       5       0.00              1        1.0             1.0        127429.56       0
165032    15689614          Hsiung          554     Spain  Female  30.0       7  161533.00              1        0.0             1.0         71173.03       0
165033    15732798         Ulyanov          850    France    Male  31.0       1       0.00              1        1.0             0.0         61581.79       1
'''
'''print(trn_csv) : index_col = 1
                id         Surname  CreditScore Geography  Gender   Age  Tenure    Balance  NumOfProducts  HasCrCard  IsActiveMember  EstimatedSalary  Exited
CustomerId
15674932         0  Okwudilichukwu          668    France    Male  33.0       3       0.00              2        1.0             0.0        181449.97       0
15749177         1   Okwudiliolisa          627    France    Male  33.0       1       0.00              2        1.0             1.0         49503.50       0
15694510         2           Hsueh          678    France    Male  40.0      10       0.00              2        1.0             0.0        184866.69       0
15741417         3             Kao          581    France    Male  34.0       2  148882.54              1        1.0             1.0         84560.88       0
15766172         4       Chiemenam          716     Spain    Male  33.0       5       0.00              2        1.0             1.0         15068.83       0
...            ...             ...          ...       ...     ...   ...     ...        ...            ...        ...             ...              ...     ...
15667085    165029            Meng          667     Spain  Female  33.0       2       0.00              1        1.0             1.0        131834.75       0
15665521    165030       Okechukwu          792    France    Male  35.0       3       0.00              1        0.0             0.0        131834.45       0
15664752    165031            Hsia          565    France    Male  31.0       5       0.00              1        1.0             1.0        127429.56       0
15689614    165032          Hsiung          554     Spain  Female  30.0       7  161533.00              1        0.0             1.0         71173.03       0
15732798    165033         Ulyanov          850    France    Male  31.0       1       0.00              1        1.0             0.0         61581.79       1
'''
'''print(trn_csv.head()) 앞에서부터 5개(Default)
            id         Surname  CreditScore Geography Gender   Age  Tenure    Balance  NumOfProducts  HasCrCard  IsActiveMember  EstimatedSalary  Exited
CustomerId
15674932     0  Okwudilichukwu          668    France   Male  33.0       3       0.00              2        1.0             0.0        181449.97       0
15749177     1   Okwudiliolisa          627    France   Male  33.0       1       0.00              2        1.0             1.0         49503.50       0
15694510     2           Hsueh          678    France   Male  40.0      10       0.00              2        1.0             0.0        184866.69       0
15741417     3             Kao          581    France   Male  34.0       2  148882.54              1        1.0             1.0         84560.88       0
15766172     4       Chiemenam          716     Spain   Male  33.0       5       0.00              2        1.0             1.0         15068.83       0
'''
'''print(trn_csv.tail()) 뒤에서부터 5개(Default)
                id    Surname  CreditScore Geography  Gender   Age  Tenure   Balance  NumOfProducts  HasCrCard  IsActiveMember  EstimatedSalary  Exited
CustomerId
15667085    165029       Meng          667     Spain  Female  33.0       2       0.0              1        1.0             1.0        131834.75       0
15665521    165030  Okechukwu          792    France    Male  35.0       3       0.0              1        0.0             0.0        131834.45       0
15664752    165031       Hsia          565    France    Male  31.0       5       0.0              1        1.0             1.0        127429.56       0
15689614    165032     Hsiung          554     Spain  Female  30.0       7  161533.0              1        0.0             1.0         71173.03       0
15732798    165033    Ulyanov          850    France    Male  31.0       1       0.0              1        1.0             0.0         61581.79       1
'''
'''print(trn_csv.isna().sum())
id                 0
Surname            0
CreditScore        0
Geography          0
Gender             0
Age                0
Tenure             0
Balance            0
NumOfProducts      0
HasCrCard          0
IsActiveMember     0
EstimatedSalary    0
Exited             0
'''



tst_csv = pd.read_csv(path + 'test.csv', index_col=0)
'''print(tst_csv) [110023 rows x 12 columns]
        CustomerId    Surname  CreditScore Geography  Gender   Age  Tenure    Balance  NumOfProducts  HasCrCard  IsActiveMember  EstimatedSalary
id
165034    15773898   Lucchese          586    France  Female  23.0       2       0.00              2        0.0             1.0        160976.75
165035    15782418       Nott          683    France  Female  46.0       2       0.00              1        1.0             0.0         72549.27
165036    15807120         K?          656    France  Female  34.0       7       0.00              2        1.0             0.0        138882.09
165037    15808905  O'Donnell          681    France    Male  36.0       8       0.00              1        1.0             0.0        113931.57
165038    15607314    Higgins          752   Germany    Male  38.0      10  121263.62              1        1.0             0.0        139431.00
...            ...        ...          ...       ...     ...   ...     ...        ...            ...        ...             ...              ...
275052    15662091      P'eng          570     Spain    Male  29.0       7  116099.82              1        1.0             1.0        148087.62
275053    15774133        Cox          575    France  Female  36.0       4  178032.53              1        1.0             1.0         42181.68
275054    15728456      Ch'iu          712    France    Male  31.0       2       0.00              2        1.0             0.0         16287.38
275055    15687541   Yegorova          709    France  Female  32.0       3       0.00              1        1.0             1.0        158816.58
275056    15663942       Tuan          621    France  Female  37.0       7   87848.39              1        1.0             0.0         24210.56
'''
'''print(tst_csv.isna().sum())
CustomerId         0
Surname            0
CreditScore        0
Geography          0
Gender             0
Age                0
Tenure             0
Balance            0
NumOfProducts      0
HasCrCard          0
IsActiveMember     0
EstimatedSalary    0
'''

sub_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# 문자 데이터의 수치화
#  CustomerId Surname  CreditScore Geography  Gender Age  Tenure  Balance  NumOfProducts  HasCrCard
#  IsActiveMember  EstimatedSalary  Exited

from sklearn.preprocessing import LabelEncoder

LE_GEO = LabelEncoder()         # class의 정의 : instance화 시킨다 >> 사용하기 쉽게 만들어 놓는다
LE_GEN = LabelEncoder()         # column의 값이 너무 크면, 함수 정의한 이후에 충돌할수도 
                                # >> 각각 

# trn_csv['Surname']
trn_csv['Geography'] = LE_GEO.fit_transform(trn_csv['Geography'])   # 특정 데이터 내 컬럼을 labeling 시켜 변환 및 적용시킨다
trn_csv['Gender'] = LE_GEN.fit_transform(trn_csv['Gender'])
tst_csv['Geography'] = LE_GEN.fit_transform(tst_csv['Geography'])
tst_csv['Gender'] = LE_GEN.fit_transform(tst_csv['Gender'])

# LE_GEN.fit(tst_csv['Gender'])
# LE_GEN.transform(tst_csv['Gender']) : 똑같은 기능인데 함수에 fit 시킨 후에 transform 진행

''' print(trn_csv['Geography'].value_counts())
0    94215
2    36213
1    34606
'''
''' print(trn_csv['Gender'].value_counts())
1    93150
0    71884
'''

trn_csv = trn_csv.drop(['CustomerId','Surname'], axis=1)
tst_csv = tst_csv.drop(['CustomerId','Surname'], axis=1)
''' print(trn_csv) [165034 rows x 11 columns]
        CreditScore  Geography  Gender   Age  Tenure    Balance  NumOfProducts  HasCrCard  IsActiveMember  EstimatedSalary  Exited
id
0               668          0       1  33.0       3       0.00              2        1.0             0.0        181449.97       0
1               627          0       1  33.0       1       0.00              2        1.0             1.0         49503.50       0
2               678          0       1  40.0      10       0.00              2        1.0             0.0        184866.69       0
3               581          0       1  34.0       2  148882.54              1        1.0             1.0         84560.88       0
4               716          2       1  33.0       5       0.00              2        1.0             1.0         15068.83       0
...             ...        ...     ...   ...     ...        ...            ...        ...             ...              ...     ...
165029          667          2       0  33.0       2       0.00              1        1.0             1.0        131834.75       0
165030          792          0       1  35.0       3       0.00              1        0.0             0.0        131834.45       0
165031          565          0       1  31.0       5       0.00              1        1.0             1.0        127429.56       0
165032          554          2       0  30.0       7  161533.00              1        0.0             1.0         71173.03       0
165033          850          0       1  31.0       1       0.00              1        1.0             0.0         61581.79       1
'''

x = trn_csv.drop(['Exited'], axis=1)
'''print(x) [165034 rows x 10 columns]
'''

y = trn_csv['Exited']
'''print(y)
id
0         0
1         0
2         0
3         0
4         0
         ..
165029    0
165030    0
165031    0
165032    0
165033    1
'''

x_trn, x_tst, y_trn, y_tst = train_test_split(x, y,
                                              train_size=0.9,
                                              shuffle=True,
                                              random_state=55)

#2. 모델
# model = RandomForestRegressor()

all_Algorithms = all_estimators(type_filter='classifier')

print('all_Algorithms :', all_Algorithms) # sklearn에서 제공하는 regressor 모두 제공
print(len(all_Algorithms))                # 55개

max_score = 0
max_name = 'Default'

for name, model in all_Algorithms:
    
    try:# 파라미터가 필요한 녀석들 발생! > 파이썬 기초 : 예외 처리
        #3. 훈련
        model = model()
        model.fit(x_trn, y_trn)
        
        #4. 평가 예측
        score = model.score(x_tst, y_tst)
        print(name, '의 정답률 :', score)
        
    except:
        print(name, ': ERROR')
        
    ####### [실습] 위치는 적절히 잘 조절해서...! except 위에도 가능 #############    
    if score > max_score:                   # print의 경우 밖에도 가능
        max_score = score
        max_name = name
        print('최고모델 :', max_name, max_score) # 최고모델 : HistGradientBoostingClassifier 0.8650024236548716
    ##########################################################################
    
'''
0.5126974582672119
DO
0.5126935243606567
CNN
0.5127053260803223
LSTM
0.41223955154418945
Conv1D
0.5128086805343628
HistGradientBoostingClassifier
0.8650024236548716
'''