# keras28_MCP_save_04_dacon_ddarung.copy
# https://dacon.io/competitions/official/235576/data
### Dropout, BatchNormalization 추가
### Scaling! MinMax
### 가중치 저장

from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score,mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd # 데이터 정리에 대한 프로그램
import numpy as np

# print(np.__version__)   # 1.23.0
# print(pd.__version__)   # 2.2.3

#1.데이터
path = './_data/dacon/따릉이/'
train_csv = pd.read_csv(path + 'train.csv', index_col = 0) 
# print(train_csv) # (1459, 11)
#                  # But. column_0은 index >> index_col로 제거
#                  # (1459, 10)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0) 
# print(test_csv)  # (715, 9)

submission_csv = pd.read_csv(path + 'submission.csv', index_col = 0)
"""
# print(submission_csv)
#                  # (715, 1)
#                  # NaN : 결칙치

# print(train_csv.shape)      # (1459, 10)
# print(test_csv.shape)       # (715, 9)
# print(submission_csv.shape) # (715, 1)

# # pandas로 가져온 파일에 대한 기능
# print(train_csv.columns)    # Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#                             #        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#                             #        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#                             #       dtype='object')
#                             # >> feature 불러오기
# print(train_csv.info())     #  0   hour                    1459 non-null   int64
#                             #  1   hour_bef_temperature    1457 non-null   float64
#                             #  2   hour_bef_precipitation  1457 non-null   float64
#                             #  3   hour_bef_windspeed      1450 non-null   float64
#                             #  4   hour_bef_humidity       1457 non-null   float64
#                             #  5   hour_bef_visibility     1457 non-null   float64
#                             #  6   hour_bef_ozone          1383 non-null   float64
#                             #  7   hour_bef_pm10           1369 non-null   float64
#                             #  8   hour_bef_pm2.5          1342 non-null   float64
#                             #  9   count                   1459 non-null   float64
#                             # >> feature 마다 데이터 갯수 >> 결칙치에 대한 정보 확인, 제거 or 예측, 제거는 애매함
#                             #                                                     >> 제거 : 데이터가 부족한거는 완성도 하락으로 직결
#                             #                                                     >> 예측 : 시간순서나, 유도리 있게 가능하면 예측 후 모델 제작
# print(train_csv.describe()) # >> 데이터의 평균, 최소, 분위 등을 제공
#               hour  hour_bef_temperature  hour_bef_precipitation  hour_bef_windspeed  ...  hour_bef_ozone  hour_bef_pm10  hour_bef_pm2.5        count
# count  1459.000000           1457.000000             1457.000000         1450.000000  ...     1383.000000    1369.000000     1342.000000  1459.000000      
# mean     11.493489             16.717433                0.031572            2.479034  ...        0.039149      57.168736       30.327124   108.563400      
# std       6.922790              5.239150                0.174917            1.378265  ...        0.019509      31.771019       14.713252    82.631733      
# min       0.000000              3.100000                0.000000            0.000000  ...        0.003000       9.000000        8.000000     1.000000      
# 25%       5.500000             12.800000                0.000000            1.400000  ...        0.025500      36.000000       20.000000    37.000000      
# 50%      11.000000             16.600000                0.000000            2.300000  ...        0.039000      51.000000       26.000000    96.000000      
# 75%      17.500000             20.100000                0.000000            3.400000  ...        0.052000      69.000000       37.000000   150.000000      
# max      23.000000             30.000000                1.000000            8.000000  ...        0.125000     269.000000       90.000000   431.000000      
# [8 rows x 10 columns]

## 결측치 처리 1. 삭제 ##
# print(train_csv.info())         # 데이터의 개수 및 특성 출력
# print(train_csv.isnull().sum()) # null 값의 모든 합을 출력
# print(train_csv.isna().sum())   # null 값의 모든 합을 출력
#     # hour                        0
#     # hour_bef_temperature        2
#     # hour_bef_precipitation      2
#     # hour_bef_windspeed          9
#     # hour_bef_humidity           2
#     # hour_bef_visibility         2
#     # hour_bef_ozone             76
#     # hour_bef_pm10              90
#     # hour_bef_pm2.5            117
#     # count                       0
# train_csv = train_csv.dropna()  # 데이터의 결측치를 삭제하고 덮어 씌워라
# print(train_csv.isnull().sum())
#     # hour                      0
#     # hour_bef_temperature      0
#     # hour_bef_precipitation    0
#     # hour_bef_windspeed        0
#     # hour_bef_humidity         0
#     # hour_bef_visibility       0
#     # hour_bef_ozone            0
#     # hour_bef_pm10             0
#     # hour_bef_pm2.5            0
#     # count                     0
# print(train_csv.info())
#     #  0   hour                    1328 non-null   int64
#     #  1   hour_bef_temperature    1328 non-null   float64
#     #  2   hour_bef_precipitation  1328 non-null   float64
#     #  3   hour_bef_windspeed      1328 non-null   float64
#     #  4   hour_bef_humidity       1328 non-null   float64
#     #  5   hour_bef_visibility     1328 non-null   float64
#     #  6   hour_bef_ozone          1328 non-null   float64
#     #  7   hour_bef_pm10           1328 non-null   float64
#     #  8   hour_bef_pm2.5          1328 non-null   float64
#     #  9   count                   1328 non-null   float64
# print(train_csv)
#     # (1328, 10)
"""

# ## 결측치 처리 2. 평균 ## 
train_csv = train_csv.fillna(train_csv.mean())
'''print(train_csv.isnull().sum())
print(train_csv.info())
'''

## 테스트의 결측치는? ## >> 제거는 절대XXXX >> 제출은 해야하니까
'''print(test_csv.info())
    #  #   Column                  Non-Null Count  Dtype
    # ---  ------                  --------------  -----
    #  0   hour                    715 non-null    int64
    #  1   hour_bef_temperature    714 non-null    float64
    #  2   hour_bef_precipitation  714 non-null    float64
    #  3   hour_bef_windspeed      714 non-null    float64
    #  4   hour_bef_humidity       714 non-null    float64
    #  5   hour_bef_visibility     714 non-null    float64
    #  6   hour_bef_ozone          680 non-null    float64
    #  7   hour_bef_pm10           678 non-null    float64
    #  8   hour_bef_pm2.5          679 non-null    float64
'''
test_csv = test_csv.fillna(test_csv.mean())
'''   # #   Column                  Non-Null Count  Dtype
    # ---  ------                  --------------  -----
    # 0   hour                    715 non-null    int64
    # 1   hour_bef_temperature    715 non-null    float64
    # 2   hour_bef_precipitation  715 non-null    float64
    # 3   hour_bef_windspeed      715 non-null    float64
    # 4   hour_bef_humidity       715 non-null    float64
    # 5   hour_bef_visibility     715 non-null    float64
    # 6   hour_bef_ozone          715 non-null    float64
    # 7   hour_bef_pm10           715 non-null    float64
    # 8   hour_bef_pm2.5          715 non-null    float64
print(test_csv.info())
'''

x = train_csv.drop(['count'], axis=1) # 앞에서 편집한 train_csv에서 count라는 axis=1 열만 짤라서 삭제
# print(x)        # (1459, 9)           # 참고로 axis = 0 은 행
y = train_csv['count']                # count 컬럼만 빼서 y로
# print(y.shape)  # (1459,)

x_trn, x_tst, y_trn, y_tst = train_test_split(x, y,
                                              test_size=0.2,
                                              random_state=7275,
                                              shuffle=True)


    ### scaling
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
# MS = MinMaxScaler()
# MS.fit(x_trn)
# x_trn = MS.transform(x_trn)
# x_tst = MS.transform(x_tst)

# AS = MaxAbsScaler()
# AS.fit(x_trn)
# x_trn = AS.transform(x_trn)
# x_tst = AS.transform(x_tst)

# SS = StandardScaler()
# SS.fit(x_trn)
# x_trn = SS.transform(x_trn)
# x_tst = SS.transform(x_tst)

RS = RobustScaler()
RS.fit(x_trn)
x_trn = RS.transform(x_trn)
x_tst = RS.transform(x_tst)

'''
RMSE : 53.944031144267946
loss : 2909.95849609375
r2 : 0.5974907928739787

[MS] 
RMSE : 41.39069578136643
loss : 1713.189697265625
r2 : 0.7630294057556429

[AS]
RMSE : 40.51574401014145
loss : 1641.5255126953125
r2 : 0.772942086943494

[SS]
RMSE : 41.04928380826973
loss : 1685.043701171875
r2 : 0.7669225970012818

[RS] ********
RMSE : 40.115434023265905
loss : 1609.248046875
r2 : 0.7774067601879184
'''

#2. 모델구성
model = Sequential()
model.add(Dense(32, input_dim=9, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(1))

epochs = 100000

    # activation : 다음 레이어로 다음 넘어갈 때 데이터를 처리
        # relu : 횟수 측정 같은 선형회귀에서 음수를 0으로 처리 시킴

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')

ES = EarlyStopping(monitor= 'val_loss',
                   mode= 'min',
                   patience= 200,
                   restore_best_weights=True)

################################# mpc 세이브 파일명 만들기 #################################
### 월일시 넣기
import datetime

path_MCP = './_save/keras28_mcp/04_dacon_ddarung/'

date = datetime.datetime.now()
# print(date)            
# print(type(date))       
date = date.strftime('%m%d_%H%M')              

# print(date)             
# print(type(date))

filename = '{epoch:04d}-{val_loss:.4f}.h5'
filepath = "".join([path_MCP,'keras28_',date, '_', filename])

MCP = ModelCheckpoint(monitor='val_loss',
                      mode='auto',
                      save_best_only=True,
                      filepath= filepath # 확장자의 경우 h5랑 같음
                                         # patience 만큼 지나기전 최저 갱신 지점        
                      )

hist = model.fit(x_trn, y_trn, epochs=epochs, batch_size=100,
          verbose=2,
          validation_split=0.2,
          callbacks=[ES, MCP])

# model.save(path + 'save_0531_0.h5')

# plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows일 경우
# plt.rcParams['axes.unicode_minus'] = False     # 마이너스 깨짐 방지

# plt.figure(figsize=(9,6)) # 9*6 사이즈 (단위 확인 피룡) # 크기설정이 1번
# plt.plot(hist.history['loss'], color='red', label='loss')
# plt.plot(hist.history['val_loss'], color='blue', label='val_loss') # y=epochs로 시간 따라서 자동으로
# plt.title('따릉이 loss') # 표 제목 한글 깨짐해결법 피룡
# plt.xlabel('epochs') # x 축 이름
# plt.ylabel('loss') # y 축 이름
# plt.legend(loc='upper right') # 우측상단 라벨표시 / defualt 는 빈자리
# plt.grid() # 격자표시


#4. 평가, 예측
loss = model.evaluate(x_tst, y_tst)
results = model.predict(x_tst)
r2 = r2_score(y_tst,results)
rmse = np.sqrt(loss)

print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print('RMSE :', rmse)
print('loss :', loss)
print('r2 :', r2)
plt.show()

'''
[MCP]
RMSE : 53.944031144267946
loss : 2909.95849609375
r2 : 0.5974907928739787
[load]

'''
# y_submit = model.predict(test_csv)
# submission_csv['count'] = y_submit
# submission_csv.to_csv(path + 'submission_0531_0.csv')