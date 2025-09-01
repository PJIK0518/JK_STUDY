# https://www.kaggle.com/c/otto-group-product-classification-challenge/data
# 41_13.copy

##########################################################################
#0. 준비
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder,OneHotEncoder, StandardScaler, OrdinalEncoder, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import time

RS = 42
##########################################################################
#1 데이터
path = 'C:/Study25/_data/kaggle/otto/'

trn_csv = pd.read_csv(path + 'train.csv', index_col=0)
tst_csv = pd.read_csv(path + 'test.csv', index_col=0)
sub_csv = pd.read_csv(path + 'sampleSubmission.csv')

""" print(trn_csv) [61878 rows x 94 columns] 
       feat_1  feat_2  feat_3  feat_4  feat_5  feat_6  feat_7  feat_8  feat_9  feat_10  feat_11  ...  feat_84  feat_85  feat_86  feat_87  feat_88  feat_89  feat_90  feat_91  feat_92  feat_93   target
id                                                                                               ...
1           1       0       0       0       0       0       0       0       0        0        1  ...        0        1        0        0        0        0        0        0        0        0  Class_1
2           0       0       0       0       0       0       0       1       0        0        0  ...        0        0        0        0        0        0        0        0        0        0  Class_1
3           0       0       0       0       0       0       0       1       0        0        0  ...        0        0        0        0        0        0        0        0        0        0  Class_1
4           1       0       0       1       6       1       5       0       0        1        1  ...       22        0        1        2        0        0        0        0        0        0  Class_1  
5           0       0       0       0       0       0       0       0       0        0        0  ...        0        1        0        0        0        0        1        0        0        0  Class_1  
...       ...     ...     ...     ...     ...     ...     ...     ...     ...      ...      ...  ...      ...      ...      ...      ...      ...      ...      ...      ...      ...      ...      ...  
61874       1       0       0       1       1       0       0       0       0        0        0  ...        0        1        0        0        0        0        0        0        2        0  Class_9  
61875       4       0       0       0       0       0       0       0       0        0        0  ...        0        0        2        0        0        2        0        0        1        0  Class_9  
61876       0       0       0       0       0       0       0       3       1        0        0  ...        0        0        3        1        0        0        0        0        0        0  Class_9  
61877       1       0       0       0       0       0       0       0       0        0        0  ...        0        0        0        0        0        1        0        3       10        0  Class_9  
61878       0       0       0       0       0       0       0       0       0        0        0  ...        0        0        0        0        0        0        0        0        2        0  Class_9  
"""
""" print(tst_csv) [144368 rows x 93 columns]
        feat_1  feat_2  feat_3  feat_4  feat_5  feat_6  feat_7  feat_8  feat_9  feat_10  feat_11  ...  feat_83  feat_84  feat_85  feat_86  feat_87  feat_88  feat_89  feat_90  feat_91  feat_92  feat_93
id                                                                                                ...
1            0       0       0       0       0       0       0       0       0        3        0  ...        0        0        0       11        1       20        0        0        0        0        0 
2            2       2      14      16       0       0       0       0       0        0        0  ...        0        0        0        0        0        0        4        0        0        2        0 
3            0       1      12       1       0       0       0       0       0        0        7  ...        0        0        0        0        0        2        0        0        0        0        1 
4            0       0       0       1       0       0       0       0       0        0        0  ...        0        0        3        1        0        0        0        0        0        0        0 
5            1       0       0       1       0       0       1       2       0        3        0  ...        1        0        0        0        0        0        0        0        9        0        0 
...        ...     ...     ...     ...     ...     ...     ...     ...     ...      ...      ...  ...      ...      ...      ...      ...      ...      ...      ...      ...      ...      ...      ... 
144364       0       0       0       0       0       0       0       0       0        1        0  ...        0        0        0        2        1        1        0        0        0        0        0 
144365       0       0       0       0       0       0       0       0       0        0        0  ...        0        0        1        4        1       11        0        0        0        0        0 
144366       0       1       0       0       0       0       1       1       0        0        0  ...        0        0        1        3        1        1        0        0        1        0        0 
144367       0       0       0       0       0       0       0       0       0        0        0  ...        0        0        0        0        0        5        0        0        0        1        0 
144368       0       0       0       0       0       0       0       0       0        0        0  ...        0        0        0        9        1        6        0        0        0        0        0  """

x = trn_csv.drop(['target'], axis=1)
y = trn_csv['target']

x_trn, x_tst, y_trn, y_tst = train_test_split(x, y,
                                                train_size = 0.5,
                                                shuffle = True,
                                                random_state = RS,
                                                stratify=y,
                                                )

""" print(x.describe())
            feat_1        feat_2        feat_3        feat_4        feat_5        feat_6        feat_7  ...       feat_87       feat_88       feat_89       feat_90       feat_91       feat_92       feat_93
count  61878.00000  61878.000000  61878.000000  61878.000000  61878.000000  61878.000000  61878.000000  ...  61878.000000  61878.000000  61878.000000  61878.000000  61878.000000  61878.000000  61878.000000
mean       0.38668      0.263066      0.901467      0.779081      0.071043      0.025696      0.193704  ...      0.393549      0.874915      0.457772      0.812421      0.264941      0.380119      0.126135
std        1.52533      1.252073      2.934818      2.788005      0.438902      0.215333      1.030102  ...      1.575455      2.115466      1.527385      4.597804      2.045646      0.982385      1.201720
min        0.00000      0.000000      0.000000      0.000000      0.000000      0.000000      0.000000  ...      0.000000      0.000000      0.000000      0.000000      0.000000      0.000000      0.000000
25%        0.00000      0.000000      0.000000      0.000000      0.000000      0.000000      0.000000  ...      0.000000      0.000000      0.000000      0.000000      0.000000      0.000000      0.000000
50%        0.00000      0.000000      0.000000      0.000000      0.000000      0.000000      0.000000  ...      0.000000      0.000000      0.000000      0.000000      0.000000      0.000000      0.000000
75%        0.00000      0.000000      0.000000      0.000000      0.000000      0.000000      0.000000  ...      0.000000      1.000000      0.000000      0.000000      0.000000      0.000000      0.000000
max       61.00000     51.000000     64.000000     70.000000     19.000000     10.000000     38.000000  ...     67.000000     30.000000     61.000000    130.000000     52.000000     19.000000     87.000000
"""
""" print(np.unique(y, return_counts=True))
(array([0, 1, 2, 3, 4, 5, 6, 7, 8]), array([ 1929, 16122,  8004,  2691,  2739, 14135,  2839,  8464,  4955],
      dtype=int64))
"""
""" print(x.shape) (61878, 93) """ 
""" print(y.shape) (61878, 9) """

#####################################
## Scaler
def Scaler(SC, x_trn, x_tst, tst_csv):
    SC.fit(x_trn)
    return SC.transform(x_trn), SC.transform(x_tst), SC.transform(tst_csv)

x_trn, x_tst,  tst_csv = Scaler(MaxAbsScaler(), x_trn, x_tst, tst_csv)

#####################################
## 증폭 : class_weight
classes = np.unique(y_trn)
CW = compute_class_weight(class_weight='balanced', classes=classes, y=y_trn)

#####################################
## Label_y
LE = LabelEncoder()
LE.fit(y_trn)
y_trn = LE.transform(y_trn)
y_tst = LE.transform(y_tst)

#####################################
## onehot_y

y_trn = y_trn.reshape(-1,1)
y_tst = y_tst.reshape(-1,1)

One = OneHotEncoder(sparse=False)
One.fit(y_trn)

y_trn = One.transform(y_trn)
y_tst = One.transform(y_tst)

# y_trn = pd.get_dummies(y_trn)
# y_tst = pd.get_dummies(y_tst)

# print(y_trn.shape) (50121, 9)
# print(y_tst.shape) (6188, 9)

### reshape
from tensorflow.keras.layers import Conv2D, Flatten

x_trn = np.array(x_trn).reshape(-1,31,3,1)
x_tst = np.array(x_tst).reshape(-1,31,3,1)
# model.add(Conv2D(10, 1, padding='same', input_shape = (31,3,1)))
# model.add(Conv2D(10, 1))
# model.add(Flatten())

##########################################################################
#2 모델구성
#####################################
# 불러오기
# path = 'C:\Study25\_data\kaggle\otto\save/'
# M = load_model(path + 'save_0606_0_2.h5')
#####################################
from tensorflow.keras.layers import Conv1D
def LT(a,b,c,DO):
        model = Sequential()
        model.add(Conv1D(10, 2, padding='same', input_shape = (31,3,1)))
        model.add(Dropout(DO))
        model.add(BatchNormalization())
        
        model.add(Conv1D(10, 2))
        model.add(Dropout(DO))
        model.add(BatchNormalization())
        
        model.add(Flatten())
        
        model.add(Dense(c, activation='relu'))
        model.add(Dropout(DO))
        model.add(BatchNormalization())
        
        model.add(Dense(c, activation='relu'))
        model.add(Dropout(DO))
        model.add(BatchNormalization())
        
        model.add(Dense(9, activation='softmax'))
        return model
    
M = LT(100,75,50, 0.3)
E, B, P, V= (100, 1000, 200, 0.1)
'''loss
loss : 0.5694356560707092
DO
loss : 0.5595657825469971
CNN
0.56212317943573
LSTM
0.39621425439093033
Conv1D
0.717155698670243
'''

#####################################
## 저장 정보
date = datetime.datetime.now()
date = date.strftime('%m%d')

saveNum = f'{date}_0_1'
fit_info = '{epochs:04d}_{val_loss:.4f}'

#####################################
## Callbacks
ES = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=P,
    restore_best_weights=False,
)

path_M = './_data/kaggle/otto/MCP/'

MCP = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    filepath="".join([path_M, 'MCP_', saveNum, '.h5'])
)

##########################################################################
#3 컴파일, 훈련
M.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

S = time.time()
H = M.fit(x_trn, y_trn,
          epochs=E, batch_size=B,
          verbose=2,
          validation_split = V,
          callbacks = [ES, MCP])
E = time.time()
T = E - S

#####################################
## 모델 및 가중치 저장
path_S = './_data/kaggle/otto/save/'
path_W = './_data/kaggle/otto/weight/'

M.save(path_S + f'save_{saveNum}.h5')
M.save_weights(path_W + f'weight_{saveNum}.h5')

##########################################################################
#4. 평가 예측
L = M.evaluate(x_tst, y_tst)
R = M.predict(x_tst)

y_pred =np.argmax(R, axis=1)
y_true =np.argmax(y_tst, axis=1)

F1 = f1_score(y_pred,y_true, average = 'macro')


#####################################
## 그래프
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(9, 6))
plt.title('OTTO')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.plot(H.history['loss'], color = 'red', label = 'loss')
plt.plot(H.history['val_loss'], color = 'green', label = 'val_loss')
plt.legend(loc = 'upper right')
plt.grid()

print('loss :', L[0])
print('ACC  :', L[1])
print('F1sc :', F1)
print('time :', T)
plt.show()

#####################################
## 파일송출
path_Sb = './_data/kaggle/otto/sub/'

y_pred = M.predict(tst_csv)
y_class = np.argmax(y_pred, axis=1)
y_sub = np.eye(9)[y_class]

sub_csv[['Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9']] = y_sub
sub_csv.to_csv(path_Sb + f'sampleSubmission_{saveNum}.csv', index=False)


