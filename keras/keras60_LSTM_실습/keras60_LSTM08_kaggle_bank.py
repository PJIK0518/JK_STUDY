### https://www.kaggle.com/competitions/playground-series-s4e1/submissions
# 41-8.copy

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

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


from tensorflow.keras.layers import Conv2D, Flatten
### reshape

x_trn = np.array(x_trn).reshape(-1,5,2)
x_tst = np.array(x_tst).reshape(-1,5,2)
# model.add(Conv2D(10, 1, padding='same', input_shape = (10,3,2)))
# model.add(Conv2D(10, 1))
# model.add(Flatten())

#2. 모델구성
from tensorflow.python.keras.layers import LSTM
def layer_tuning(a,b,d,c,e) :
    
    model = Sequential()
    model.add(LSTM(10, input_shape = (5,2)))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dropout(0.1))
    model.add(Dense(c, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(c, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(d, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(d, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(e, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))
    
    return model

M = layer_tuning(64,512,256,128,64)
E = 100000
B = 5000
V = 0.3
P = 100
'''loss
0.5126974582672119
DO
0.5126935243606567
CNN
0.5127053260803223
LSTM
0.41223955154418945
'''

# ES = EarlyStopping(monitor='val_loss',
#                    mode='min',
#                    patience=P,
#                    restore_best_weights=True)

#3. 컴파일 훈련
M.compile(loss='binary_crossentropy',
          optimizer='adam',
          metrics=['acc'])

################################# mpc 세이브 파일명 만들기 #################################
### 월일시 넣기
# import datetime

# path_MCP = './_save/keras28_mcp/08_kaggle_bank/'

# date = datetime.datetime.now()
# # print(date)            
# # print(type(date))       
# date = date.strftime('%m%d_%H%M')              

# # print(date)             
# # print(type(date))

# filename = '{epoch:04d}-{val_loss:.4f}.h5'
# filepath = "".join([path_MCP,'keras28_',date, '_', filename])

# MCP = ModelCheckpoint(monitor='val_loss',
#                       mode='auto',
#                       save_best_only=True,
#                       filepath= filepath # 확장자의 경우 h5랑 같음
#                                          # patience 만큼 지나기전 최저 갱신 지점        
#                       )
import time
start = time.time()
H = M.fit(x_trn, y_trn,
      epochs = 100, batch_size = 32,
      verbose = 2,
      validation_split = V,
    #   callbacks = [ES, MCP]
      )
end = time.time()

'''
GPU 있다!!!
소요시간 : 1661.1106805801392

GPU 없다...
소요시간 : 762.5122427940369
'''


# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.figure(figsize= (9,6))
# plt.title('BANK')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.plot(H.history['loss'], color = 'red', label='loss')
# plt.plot(H.history['val_loss'], color = 'green', label='val_loss')
# plt.legend(loc = 'upper right')
# plt.grid()

# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.figure(figsize= (9,6))
# plt.title('BANK')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.plot(H.history['acc'], color = 'red', label='acc')
# plt.legend(loc = 'upper right')
# plt.grid()


#4. 평가 예측
loss = M.evaluate(x_tst,y_tst)
print(loss)
# import tensorflow as tf
# gpus = tf.config.list_physical_devices('GPU')
# print(gpus)

# if gpus:
#     print('GPU 있다!!!')
# else:
#     print('GPU 없다...')

# time = end - start
# print("소요시간 :", time)
### 파일 송출

# y_sub = M.predict(tst_csv)
# sub_csv['Exited'] = y_sub
# sub_csv.to_csv(path + 'sample_submission_0527_2.csv')