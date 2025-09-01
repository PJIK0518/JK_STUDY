# https://dacon.io/competitions/official/236068/overview/description
# 33-7.copy
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#1. 데이터
path = './_data/dacon/당뇨병/'

trn_csv = pd.read_csv(path + "train.csv", index_col=0)
'''print(trn_csv) [652 rows x 9 columns]
           Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  DiabetesPedigreeFunction  Age  Outcome
ID
TRAIN_000            4      103             60             33      192  24.0                     0.966   33        0
TRAIN_001           10      133             68              0        0  27.0                     0.245   36        0
TRAIN_002            4      112             78             40        0  39.4                     0.236   38        0
TRAIN_004            1      114             66             36      200  38.1                     0.289   21        0
TRAIN_647            1       91             64             24        0  29.2                     0.192   21        0
TRAIN_648           10      122             68              0        0  31.2                     0.258   41        0
TRAIN_649            8       84             74             31        0  38.3                     0.457   39        0
TRAIN_650            2       81             72             15       76  30.1                     0.547   25        0
TRAIN_651            1      107             68             19        0  26.5                     0.165   24        0
'''
'''print(trn_csv.info())
 #   Column                    Non-Null Count  Dtype
---  ------                    --------------  -----
 0   Pregnancies               652 non-null    int64
 1   Glucose                   652 non-null    int64
 2   BloodPressure             652 non-null    int64
 3   SkinThickness             652 non-null    int64
 4   Insulin                   652 non-null    int64
 5   BMI                       652 non-null    float64
 6   DiabetesPedigreeFunction  652 non-null    float64
 7   Age                       652 non-null    int64
 8   Outcome                   652 non-null    int64
'''

tst_csv = pd.read_csv(path + "test.csv", index_col=0)
'''print(tst_csv) [116 rows x 8 columns]
           ID  Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  DiabetesPedigreeFunction  Age
0    TEST_000            5      112             66              0        0  37.8                     0.261   41
1    TEST_001            3      107             62             13       48  22.9                     0.678   23
2    TEST_002            3      113             44             13        0  22.4                     0.140   22
4    TEST_004            1      107             72             30       82  30.8                     0.821   24
111  TEST_111           10      111             70             27        0  27.5                     0.141   40
112  TEST_112            1      119             54             13       50  22.3                     0.205   24
113  TEST_113            3      187             70             22      200  36.4                     0.408   36
114  TEST_114            3      100             68             23       81  31.6                     0.949   28
115  TEST_115            2       84              0              0        0   0.0                     0.304   21
'''
'''print(tst_csv.info())
 #   Column                    Non-Null Count  Dtype
---  ------                    --------------  -----
 0   Pregnancies               116 non-null    int64
 1   Glucose                   116 non-null    int64
 2   BloodPressure             116 non-null    int64
 3   SkinThickness             116 non-null    int64
 4   Insulin                   116 non-null    int64
 5   BMI                       116 non-null    float64
 6   DiabetesPedigreeFunction  116 non-null    float64
 7   Age                       116 non-null    int64
'''

sub_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)
''' print(sub_csv) [116 rows x 1 columns]
          Outcome
ID
TEST_000        0
TEST_001        0
TEST_002        0
TEST_003        0
TEST_004        0
...           ...
TEST_111        0
TEST_112        0
TEST_113        0
TEST_114        0
TEST_115        0
'''

## nan_null / 0이 nan 값, 대체!

x = trn_csv.drop(['Outcome'], axis=1)
y = trn_csv['Outcome']

x = x.replace(0, np.nan)
x = x.fillna(x.mean())
# print(x.info())

tst_csv = tst_csv.replace(0, np.nan)
tst_csv = tst_csv.fillna(tst_csv.mean())
# print(tst_csv.info())

# print(x.shape, y.shape) (652, 8) (652,)

x_trn, x_tst, y_trn, y_tst = train_test_split(x, y,
                                              train_size=0.9,
                                              shuffle=True,
                                              random_state=55)

# [['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']]

C = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']

MS = MinMaxScaler()

MS.fit(x_trn[C])
x_trn[C] = MS.transform(x_trn[C])
x_tst[C] = MS.transform(x_tst[C])
tst_csv[C] = MS.transform(tst_csv[C])

from tensorflow.keras.layers import Conv2D, Flatten
### reshape

x_trn = np.array(x_trn).reshape(-1,2,2,2)
x_tst = np.array(x_tst).reshape(-1,2,2,2)
# model.add(Conv2D(10, 1, padding='same', input_shape = (10,3,2)))
# model.add(Conv2D(10, 1))
# model.add(Flatten())

#2. 모델구성
def layer_tuning(a,b) :
    
    model = Sequential()
    model.add(Conv2D(10, 1, padding='same', input_shape = (2,2,2)))
    model.add(Dropout(0.1))
    model.add(Conv2D(10, 1))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    
    return model


M = layer_tuning(10,3)
# E = 100000
# B = 50
# V = 0.2
# P = 200

''' loss
0.5406605005264282
DO
0.4480418264865875
CNN
0.43065935373306274
'''

# ES = EarlyStopping(monitor='val_loss',
#                    mode='min',
#                    patience=P,
#                    restore_best_weights=True)

################################# mpc 세이브 파일명 만들기 #################################
### 월일시 넣기
# import datetime

# path_MCP = './_save/keras28_mcp/07_dacon_당뇨병/'

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

#3. 컴파일 훈련
M.compile(loss='binary_crossentropy',
          optimizer='adam',
          metrics=['acc'])
import time
start = time.time()
H = M.fit(x_trn, y_trn,
        epochs=100,
        batch_size=32,
        verbose=2,
        validation_split=0.2,
        # callbacks=[ES, MCP]
        )
end = time.time()

# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.figure(figsize=(9,6))
# plt.title('Dacon 당뇨병')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.plot(H.history['loss'], color = "red", label='loss')
# plt.plot(H.history['val_loss'], color = "green", label='val_loss')
# plt.legend(loc = 'upper right')
# plt.grid()

# plt.figure(figsize=(9,6))
# plt.title('Dacon 당뇨병')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.plot(H.history['acc'], color = "red", label='acc')
# plt.plot(H.history['val_acc'], color = "green", label='val_acc')
# plt.legend(loc = 'upper right')
# plt.grid()

#4. 평가 예측
loss = M.evaluate(x_tst, y_tst)
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

'''
GPU 있다!!!
소요시간 : 6.58752965927124

GPU 없다...
소요시간 : 2.7388384342193604
'''
# plt.show()

### 파일 제출
# y_sub = M.predict(tst_csv)
# y_sub = np.round(y_sub)
# sub_csv['Outcome'] = y_sub
# sub_csv.to_csv(path + 'sample_submission_0528_1.csv')
