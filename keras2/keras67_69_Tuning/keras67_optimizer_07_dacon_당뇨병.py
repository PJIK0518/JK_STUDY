# https://dacon.io/competitions/official/236068/overview/description
# 31-7.copy
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#1. 데이터
path = './Study25/_data/dacon/00 당뇨병/'

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


#3. 컴파일, 훈련
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, Adagrad, SGD, RMSprop

optim = [Adam, Adagrad, SGD, RMSprop]
lrlst = [0.1, 0.01, 0.05, 0.001, 0.0001]

best_op = ''
best_lr = 0
best_sc = -1000

for idx_1, op in enumerate(optim):
    for idx_2, lr in enumerate(lrlst):
        #2. 모델구성
        model = Sequential()
        model.add(Dense(32, input_dim=8, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(1, activation = 'sigmoid'))

        epochs = 100000
        optimizer_instance = op(learning_rate=lr)
        model.compile(loss = 'mse', optimizer =optimizer_instance)
                
        ES = EarlyStopping(monitor='val_loss',
                        mode= 'min',
                        patience= 100,
                        restore_best_weights= True)


        hist = model.fit(x_trn, y_trn, epochs = 100, batch_size = 32,
                verbose=0,
                validation_split=0.2)

        loss = model.evaluate(x_tst, y_tst)
        results = np.round(model.predict([x_tst]))

        ACC = accuracy_score(y_tst, results)
        print(f'⏩⏩  {idx_1}/{len(optim)} | {idx_2}/{len(lrlst)}  ⏩⏩')
        print('optimizer    :', op)
        print('learing_rate :', lr)
        print('score        :',ACC)
        if ACC >= best_sc:
            best_op = op
            best_lr = lr
            best_sc = ACC

print('✅  DONE ✅')
print('최종옵티머 :', best_op)
print('최종학습률 :', best_lr)
print('최종모델능 :', best_sc)

# ✅  DONE ✅
# 최종옵티머 : <class 'keras.src.optimizers.rmsprop.RMSprop'>
# 최종학습률 : 0.05
# 최종모델능 : 0.8939393939393939