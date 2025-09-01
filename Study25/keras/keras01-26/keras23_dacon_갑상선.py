# https://dacon.io/competitions/official/236488/overview/description
# 연습 Dacon 입문자용

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow_addons.metrics import F1Score
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler, StandardScaler, RobustScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time 

RS = 190

#1. 데이터
path = 'C:/Study25/_data/dacon/갑상선/'
trn_csv = pd.read_csv(path + 'train.csv', index_col=0)
tst_csv = pd.read_csv(path + 'test.csv', index_col=0)
sub_csv = pd.read_csv(path + 'sample_submission.csv')

trn_csv = trn_csv.drop(['T3_Result', 'Age', 'Nodule_Size', 'T4_Result', 'TSH_Result'], axis=1)
tst_csv = tst_csv.drop(['T3_Result', 'Age', 'Nodule_Size', 'T4_Result', 'TSH_Result'], axis=1)



'''print(trn_csv) [87159 rows x 15 columns]
             Age Gender Country Race Family_Background Radiation_History Iodine_Deficiency       Smoke Weight_Risk Diabetes  Nodule_Size  TSH_Result  T4_Result  T3_Result  Cancer
ID
TRAIN_00000   80      M     CHN  ASN          Positive           Exposed        Sufficient  Non-Smoker   Not Obese       No     0.650355    2.784735   6.744603   2.575820       1
TRAIN_00001   37      M     NGA  ASN          Positive         Unexposed        Sufficient      Smoker       Obese       No     2.950430    0.911624   7.303305   2.505317       1
TRAIN_00002   71      M     CHN  MDE          Positive         Unexposed        Sufficient  Non-Smoker   Not Obese      Yes     2.200023    0.717754  11.137459   2.381080       0
TRAIN_00003   40      F     IND  HSP          Negative         Unexposed        Sufficient  Non-Smoker       Obese       No     3.370796    6.846380  10.175254   0.753023       0
TRAIN_00004   53      F     CHN  CAU          Negative         Unexposed        Sufficient  Non-Smoker   Not Obese       No     4.230048    0.439519   7.194450   0.569356       1
...          ...    ...     ...  ...               ...               ...               ...         ...         ...      ...          ...         ...        ...        ...     ...
TRAIN_87154   65      F     IND  ASN          Positive         Unexposed        Sufficient  Non-Smoker   Not Obese       No     0.510802    3.786859   4.838150   0.625754       1
TRAIN_87155   53      M     NGA  ASN          Negative         Unexposed        Sufficient  Non-Smoker       Obese       No     0.980413    4.335395   8.937716   2.728584       0
TRAIN_87156   29      F     RUS  CAU          Negative         Unexposed        Sufficient  Non-Smoker   Not Obese       No     0.180998    5.724924   4.847265   3.318609       0
TRAIN_87157   52      F     IND  ASN          Positive         Unexposed        Sufficient  Non-Smoker   Not Obese       No     2.420773    4.978069  10.867191   2.259199       1
TRAIN_87158   81      F     USA  CAU          Negative           Exposed        Sufficient  Non-Smoker   Not Obese       No     4.800598    1.353310  11.505252   3.335156       0
'''
'''print(trn_csv.columns)
['Age', 'Gender', 'Country', 'Race', 'Family_Background','Radiation_History', 'Iodine_Deficiency', 'Smoke', 'Weight_Risk', 'Diabetes', 'Nodule_Size', 'TSH_Result', 'T4_Result', 'T3_Result', 'Cancer']
'''

# ### LabelEncoder : ['Gender  Country  Race  Family_Background  Radiation_History   Iodine_Deficiency  Smoke  Weight_Risk  Diabetes ]
OE = OrdinalEncoder()
OE_col = [ 'Gender', 'Family_Background','Radiation_History', 'Iodine_Deficiency', 'Smoke', 'Weight_Risk', 'Diabetes']

OE.fit(trn_csv[OE_col])
trn_csv[OE_col] = OE.transform(trn_csv[OE_col])
tst_csv[OE_col] = OE.transform(tst_csv[OE_col])

### OneHotEncoder for x

ONE_x = OneHotEncoder()
ONE_col = ['Race','Country']

one_encoded = ONE_x.fit_transform(trn_csv[ONE_col]).toarray()

one_encoded_df = pd.DataFrame(one_encoded,
                              columns=ONE_x.get_feature_names_out(ONE_col),
                              index=trn_csv.index)

trn_csv = trn_csv.drop(ONE_col, axis=1)
trn_csv = pd.concat([trn_csv, one_encoded_df], axis=1)

tst_encoded = ONE_x.transform(tst_csv[ONE_col]).toarray()
tst_encoded_df = pd.DataFrame(tst_encoded,
                              columns=ONE_x.get_feature_names_out(ONE_col),
                              index=tst_csv.index)

tst_csv = tst_csv.drop(ONE_col, axis=1)
tst_csv = pd.concat([tst_csv, tst_encoded_df], axis=1)

# ['T3_Result', 'Age', 'Nodule_Size', 'T4_Result', 'TSH_Result']

x = trn_csv.drop(['Cancer'], axis=1)
y = trn_csv['Cancer']

RSc = RobustScaler()
RSc.fit(x)
x = RSc.transform(x)

### 치중 데이터 증폭
# ros = RandomOverSampler(random_state=RS)
# x, y = ros.fit_resample(x, y)

x_trn, x_tst, y_trn, y_tst = train_test_split(x,  y,
                                              train_size=0.7,
                                              shuffle=True,
                                              random_state=RS,
                                              stratify=y
                                              )

x_trn, x_val, y_trn, y_val = train_test_split(x_trn,  y_trn,
                                              train_size=0.9,
                                              shuffle=True,
                                              random_state=RS,
                                              stratify=y_trn
                                              )

smt = SMOTE(random_state=RS)
x_trn, y_trn = smt.fit_resample(x_trn, y_trn)

## Scaling
# SS = StandardScaler()
# SS_col = ['Nodule_Size', 'TSH_Result', 'T4_Result', 'T3_Result']

# SS.fit(x_trn[SS_col])
# x_trn[SS_col] = SS.transform(x_trn[SS_col])
# x_tst[SS_col] = SS.transform(x_tst[SS_col])
# tst_csv[SS_col] = SS.transform(tst_csv[SS_col])

# MS = MinMaxScaler()
# MS_col = ['Age']

# MS.fit(x_trn[MS_col])
# x_trn[MS_col] = MS.transform(x_trn[MS_col])
# x_tst[MS_col] = MS.transform(x_tst[MS_col])
# tst_csv[MS_col] = MS.transform(tst_csv[MS_col])
'''
print(x_trn, y) : SS 적용시 음수값 // 결과보고 생각
print(x.shape, y.shape) (87159, 14) (87159,) '''

#2. 모델구성
# M = load_model(path + 'save_0603_4.h5')
def layer_tuning(a,DO):
    model = Sequential()
    model.add(Dense(a*4, input_dim=22, activation='relu'))
    model.add(Dropout(DO))
    model.add(BatchNormalization())
    
    model.add(Dense(a*2, activation='relu'))
    model.add(Dropout(DO))
    model.add(BatchNormalization())
    
    model.add(Dense(a, activation='relu'))
    model.add(Dropout(DO))
    model.add(BatchNormalization())
    
    model.add(Dense(a, activation='relu'))
    model.add(Dropout(DO))
    model.add(BatchNormalization())
    
    model.add(Dense(1, activation='sigmoid'))
    
    return model
  
M = layer_tuning(28,0.2)
E = 100000
P = 300
B = 2250
saveNum = "0604_3"

M.load_weights(path + 'weight_0604_2.h5')

ES = EarlyStopping(monitor ='val_f1_score',
                   mode = 'max',
                   patience= P,
                   restore_best_weights=True)

MCP = ModelCheckpoint(monitor='val_f1_score',
                      mode='max',
                      save_best_only=True,
                      filepath = path + '/MCP/' + f'MCP_{saveNum}.h5')

#3. 컴파일 훈련
M.compile(loss = 'binary_crossentropy',
          optimizer='adam',
          metrics=['acc', 
                   F1Score(num_classes=1, threshold=0.5, name='f1_score')
                   ])

H = M.fit(x_trn, y_trn,
          epochs = E, batch_size=B,
          verbose = 2,
          validation_data=(x_val, y_val),
          callbacks=[ES, MCP])

M.save(path + f'save_{saveNum}.h5')
M.save_weights(path + f'weight_{saveNum}.h5')

# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.figure(figsize=(9,6))
# plt.title('갑상선')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.plot(H.history['loss'], color = 'red', label = 'loss')
# plt.plot(H.history['val_loss'], color = 'green', label = 'val_loss')
# plt.legend(loc = 'upper right')
# plt.grid()

# plt.figure(figsize=(9,6))
# plt.title('갑상선')
# plt.xlabel('epochs')
# plt.ylabel('f1_score')
# plt.plot(H.history['f1_score'], color = 'red', label = 'f1_score')
# plt.plot(H.history['val_f1_score'], color = 'green', label = 'val_f1_score')
# plt.legend(loc = 'lower right')
# plt.grid()

#4. 평가 예측
LSAC = M.evaluate(x_tst, y_tst)
R = M.predict(x_tst)
R = np.round(R)
F1 = f1_score(y_tst,R)

print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print('loss :', LSAC[0])
print('acc  :', LSAC[1])
print('f1_sc:', F1)

""" R = pd.Series(R.flatten())
print(pd.value_counts(y_tst))
print(pd.value_counts(R))
y_tst.to_csv(path + 'y_tst_confrim.csv', index=False)
R.to_csv(path + 'R_confrim.csv', index=False)
 """
# plt.show()

######
y_sub = M.predict(tst_csv)
y_sub = np.round(y_sub)
sub_csv['Cancer'] = y_sub
sub_csv.to_csv(path + f'sample_submission_{saveNum}.csv', index=False)