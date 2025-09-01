# https://dacon.io/competitions/official/235576/data
# 대회에서 제공하는 데이터 train : 모델구성, 훈련 및 평가용 데이터
                        # test : y 값을 구해야하는 데이터
                        # submission : y값 넣어서 제출하는 데이터

import numpy as np
import pandas as pd # 데이터 정리에 대한 프로그램

print(np.__version__)   # 1.23.0
print(pd.__version__)   # 2.2.3

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error

#1.데이터
path = './_data/dacon/따릉이/'
                 # python 변수 설정 : 문자 + 문자 >>> 문자문자
train_csv = pd.read_csv(path + 'train.csv', index_col = 0) 
#                  # . : 현재위치, .. : 상위폴더, / : 하위폴더
#                  # index_col = n : (n-1) 번째 column을 index로 인식
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
                                              random_state=38,
                                              shuffle=True)

#2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=9, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
epochs = 300

    # activation : 다음 레이어로 다음 넘어갈 때 데이터를 처리
        # relu : 횟수 측정 같은 선형회귀에서 음수를 0으로 처리 시킴

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_trn, y_trn, epochs=epochs, batch_size=32)

#4. 평가, 예측
loss = model.evaluate(x_tst, y_tst)
results = model.predict(x_tst)
r2 = r2_score(y_tst,results)
rmse = np.sqrt(loss)

print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print('RMSE :', rmse)
print('loss :', loss)
print('r2 :', r2)

""" 결측치(평균) / train_size : 0.9 / epochs : 100 / batch_size : 32 / hidden_layer : 64 256 128 64 / random_state : 123
loss : 2384.44140625
r2 : 0.5999434571806153
loss : 2931.406494140625
r2 : 0.5081747461220845
loss : 2529.7392578125
r2 : 0.5755656129076987
loss : 2434.43603515625
r2 : 0.5915553796477331
loss : 3244.729736328125
r2 : 0.45560600664693407
"""
""" 결측치(평균) / train_size : 0.9 / epochs : 200 / batch_size : 32 / hidden_layer : 64 256 128 64 / random_state : 123
loss : 2429.74658203125
r2 : 0.5923422413950672
loss : 2461.34130859375
r2 : 0.5870412570828072
loss : 2705.369140625
r2 : 0.5460987732983515
loss : 2589.19384765625
r2 : 0.5655904478928085
loss : 2707.94873046875
r2 : 0.545665975117653
"""
""" 결측치(평균) / train_size : 0.9 / epochs : 500 / batch_size : 32 / hidden_layer : 64 256 128 64 / random_state : 123
loss : 2513.592529296875
r2 : 0.578274718637497
loss : 2441.556884765625
r2 : 0.5903607112920032
loss : 2427.2392578125
r2 : 0.5927628605643462
loss : 2659.67626953125
r2 : 0.5537650453054292
loss : 2531.913818359375
r2 : 0.5752008225950418
"""
""" 결측치(평균) / train_size : 0.9 / epochs : 1000 / batch_size : 50 / hidden_layer : 64 256 128 64 / random_state : 123
loss : 2458.108642578125
r2 : 0.5875836535218784
RMSE : 50.12179014816061
loss : 2512.19384765625
r2 : 0.5785093512728179
"""
""" 결측치(평균) / train_size : 0.9 / epochs : 1000 / batch_size : 32 / hidden_layer : 64 128 128 64 / random_state : 123
RMSE : 49.339148376314725
loss : 2434.3515625
r2 : 0.591569551406774
"""
""" 결측치(평균) / train_size : 0.9 / epochs : 1000 / batch_size : 50 / hidden_layer : 50 150 100 50 / random_state : 123
RMSE : 51.61263660977319    49.75992167083365
loss : 2663.8642578125      2476.0498046875
r2 : 0.5530623984613883     0.5845735084049789
"""
""" 결측치(평균) / train_size : 0.9 / epochs : 1000 / batch_size : 32 / hidden_layer : 50 200 150 100 50 / random_state : 123
RMSE : 49.314363973802045
loss : 2431.906494140625
r2 : 0.5919797772775581
""" 
""" 결측치(평균) / train_size : 0.9 / epochs : 1000 / batch_size : 16 / hidden_layer : 50 200 150 100 50 / random_state : 123
RMSE : 49.60436490449328
loss : 2460.593017578125
r2 : 0.5871668903842637
결측치(평균) / train_size : 0.9 / epochs : 500 / batch_size : 16 / hidden_layer : 50 200 150 100 50 / random_state : 123
RMSE : 51.8588665612316
loss : 2689.342041015625
r2 : 0.548787817229525
결측치(평균) / train_size : 0.9 / epochs : 500 / batch_size : 32 / hidden_layer : 50 150 150 100 50 / random_state : 123
RMSE : 50.50748561862514
loss : 2551.006103515625
r2 : 0.5719975837323369
"""
""" 결측치(평균) / train_size : 0.9 / epochs : 1000 / batch_size : 32 / hidden_layer : 50 150 100 50 / random_state : 4
RMSE : 48.45977310486503    48.30283952978981   48.928599126686635
loss : 2348.349609375       2333.164306640625   2394.0078125
r2 : 0.5741094799839672     0.5768634051301068  0.5658290722432836

결측치(평균) / train_size : 0.9 / epochs : 1000 / batch_size : 16 / hidden_layer : 50 150 100 50 / random_state : 123
RMSE : 48.627430048533306   49.47138041806343
loss : 2364.626953125       2447.41748046875
r2 : 0.5711574682774887     0.5893773434997116

결측치(평균) / train_size : 0.9 / epochs : 1000 / batch_size : 32 / hidden_layer : 50 200 150 100 / random_state : 123
RMSE : 49.46032481159773
loss : 2446.32373046875
r2 : 0.5895609724676126

결측치(평균) / train_size : 0.9 / epochs : 1000 / batch_size : 32 / hidden_layer : 50 250 200 150 100 50 / random_state : 123
RMSE : 50.27237579007108
loss : 2527.311767578125
r2 : 0.5759728686056191

결측치(평균) / train_size : 0.9 / epochs : 500 / batch_size : 32 / hidden_layer : 50 150 100 50 / random_state : 123
RMSE : 49.4494173175668
loss : 2445.244873046875
r2 : 0.5897419425783035

결측치(평균) / train_size : 0.9 / epochs : 500 / batch_size : 16 / hidden_layer : 50 150 100 50 / random_state : 123
RMSE : 49.50214049980566    
loss : 2450.4619140625      
r2 : 0.5888666812470588     

결측치(평균) / train_size : 0.9 / epochs : 300 / batch_size : 16 / hidden_layer : 50 150 100 50 / random_state : 123
RMSE : 49.80124559185242    
loss : 2480.1640625         
r2 : 0.5838832454206946

결측치(평균) / train_size : 0.9 / epochs : 200 / batch_size : 16 / hidden_layer : 50 150 100 50 / random_state : 123
RMSE : 51.52166470130551
loss : 2654.48193359375
r2 : 0.5546365795865615

결측치(평균) / train_size : 0.9 / epochs : 100 / batch_size : 16 / hidden_layer : 50 150 100 50 / random_state : 123
RMSE : 52.097039863124664
loss : 2714.1015625
r2 : 0.5446337285067835
"""
""" 결측치(평균) / train_size : 0.9 / epochs : 400 / batch_size : 16 / hidden_layer : 50 150 100 50 / random_state : 123
RMSE : 49.18166261423052    49.22851589621355
loss : 2418.8359375         2423.44677734375
r2 : 0.5941727513511739     0.5933991689790508

결측치(평균) / train_size : 0.9 / epochs : 400 / batch_size : 16 / hidden_layer : 50 150 100 50 / random_state : 123 / relu
RMSE : 49.18637824324434
loss : 2419.2998046875
r2 : 0.5940949362229271

결측치(평균) / train_size : 0.9 / epochs : 400 / batch_size : 32 / hidden_layer : 50 150 100 50 / random_state : 123 / relu
RMSE : 50.17545874385455
loss : 2517.57666015625
r2 : 0.5776062337189526

결측치(평균) / train_size : 0.9 / epochs : 1000 / batch_size : 32 / hidden_layer : 50 150 100 50 / random_state : 123 / relu
RMSE : 48.409039017457786
loss : 2343.43505859375
r2 : 0.6068233662044865

결측치(평균) / train_size : 0.9 / epochs : 1000 / batch_size : 50 / hidden_layer : 50 150 100 50 / random_state : 123 / relu
RMSE : 46.13061555416587
loss : 2128.03369140625
r2 : 0.6429629398184828

결측치(평균) / train_size : 0.9 / epochs : 2000 / batch_size : 50 / hidden_layer : 50 150 100 50 / random_state : 123 / relu
RMSE : 42.80473669348332
loss : 1832.2454833984375
r2 : 0.6925896781623075

결측치(평균) / train_size : 0.9 / epochs : 5000 / batch_size : 50 / hidden_layer : 50 150 100 50 / random_state : 123 / relu
RMSE : 46.23838008590645
loss : 2137.98779296875
r2 : 0.6412928699982853

결측치(평균) / train_size : 0.9 / epochs : 3500 / batch_size : 50 / hidden_layer : 50 150 100 50 / random_state : 123 / relu
RMSE : 45.13222760677784
loss : 2036.91796875
r2 : 0.6582501837777872

결측치(평균) / train_size : 0.9 / epochs : 2500 / batch_size : 50 / hidden_layer : 50 150 100 50 / random_state : 123 / relu
RMSE : 45.78940040504052
loss : 2096.669189453125
r2 : 0.6482252333983622

결측치(평균) / train_size : 0.9 / epochs : 2000 / batch_size : 50 / hidden_layer : 50 150 100 50 / random_state : 123 / relu
RMSE : 54.82307525192403
loss : 3005.569580078125
r2 : 0.4957318614970926

결측치(평균) / train_size : 0.9 / epochs : 2000 / batch_size : 32 / hidden_layer : 50 150 100 50 / random_state : 123 / relu
RMSE : 43.795431210594046
loss : 1918.039794921875
r2 : 0.6781952917350107
"""
""" 결측치(평균) / train_size : 0.9 / epochs : 2000 / batch_size : 16 / hidden_layer : 50 150 100 50 / random_state : 123 / relu
RMSE : 42.12348876415702
loss : 1774.3883056640625
r2 : 0.7022968536569999

결측치(평균) / train_size : 0.9 / epochs : 2000 / batch_size : 16 / hidden_layer : 50 150 100 50 / random_state : 7 / relu
RMSE : 45.033028243862304
loss : 2027.9736328125
r2 : 0.7352441674625121

결측치(평균) / train_size : 0.9 / epochs : 2000 / batch_size : 16 / hidden_layer : 50 150 100 50 / random_state : 5702 / relu
RMSE : 43.371954819257915
loss : 1881.12646484375
r2 : 0.7571574484973652
"""
""" 결측치(평균) / train_size : 0.9 / epochs : 2000 / batch_size : 16 / hidden_layer : 50 150 100 50 / random_state : 915 / relu
RMSE : 38.53379072118068    36.92470167448885
loss : 1484.85302734375     1363.43359375
r2 : 0.7932195721395716     0.8101284394710754

결측치(평균) / train_size : 0.9 / epochs : 2000 / batch_size : 32 / hidden_layer : 50 150 100 50 / random_state : 915 / relu
RMSE : 40.409269451664024
loss : 1632.9090576171875
r2 : 0.7726012932136879

결측치(평균) / train_size : 0.9 / epochs : 2000 / batch_size : 10 / hidden_layer : 50 150 100 50 / random_state : 915 / relu
RMSE : 45.183801607814885
loss : 2041.575927734375
r2 : 0.715690391943105

결측치(평균) / train_size : 0.9 / epochs : 2000 / batch_size : 20 / hidden_layer : 50 150 100 50 / random_state : 915 / relu
RMSE : 42.10436977442558
loss : 1772.7779541015625
r2 : 0.7531231622215702

결측치(평균) / train_size : 0.9 / epochs : 2000 / batch_size : 16 / hidden_layer : 50 150 100 50 / random_state : 915 / relu
RMSE : 42.921879266017264
loss : 1842.2877197265625
r2 : 0.7434432826075661

결측치(평균) / train_size : 0.9 / epochs : 2000 / batch_size : 16 / hidden_layer : 50 150 100 50 / random_state : 264 / relu
RMSE : 43.36781449398131
loss : 1880.767333984375
r2 : 0.7369792908756316

결측치(평균) / train_size : 0.9 / epochs : 2000 / batch_size : 16 / hidden_layer : 400 400 200 200 / random_state : 264 / relu
RMSE : 41.398343879635604
loss : 1713.8228759765625
r2 : 0.7603260686185935
"""
""" 결측치(평균) / train_size : 0.9 / epochs : 2000 / batch_size : 16 / hidden_layer : 50 150 100 50 / random_state : 915 / relu
결측치(평균) / train_size : 0.9 / epochs : 2000 / batch_size : 16 / hidden_layer : 150 100 50 50 / random_state : 915 / relu
RMSE : 40.612645356965174
loss : 1649.386962890625
r2 : 0.7703065518559036

결측치(평균) / train_size : 0.9 / epochs : 2000 / batch_size : 16 / hidden_layer : 50 150 100 50 / random_state : 4 / relu
RMSE : 45.50491205885772
loss : 2070.697021484375
r2 : 0.6244638489512334

결측치(평균) / train_size : 0.9 / epochs : 2000 / batch_size : 16 / hidden_layer : 50 150 100 50 / random_state : 5772 / relu
RMSE : 37.09644676117495
loss : 1376.1463623046875
r2 : 0.7824333390715016

결측치(평균) / train_size : 0.9 / epochs : 2000 / batch_size : 16 / hidden_layer : 16 32 64 128 64 32 16 8 4 2 / random_state : 5772 / relu
RMSE : 39.47953169493734
loss : 1558.6334228515625
r2 : 0.7535824042932495

결측치(평균) / train_size : 0.9 / epochs : 2000 / batch_size : 16 / hidden_layer : 64 64 64 64 64 64 / random_state : 5772 / relu
RMSE : 40.035159108510484
loss : 1602.81396484375
r2 : 0.7465975535602762

결측치(평균) / train_size : 0.9 / epochs : 2000 / batch_size : 16 / hidden_layer : 64 64 64 64 64 64 64 / random_state : 5772 / relu
RMSE : 40.896141441040925
loss : 1672.494384765625
r2 : 0.7355811465216079

결측치(평균) / train_size : 0.9 / epochs : 2000 / batch_size : 16 / hidden_layer : 64 64 64 64 64 64 64 64 / random_state : 5772 / relu
RMSE : 52.40497119414698
loss : 2746.281005859375
r2 : 0.5658171345162598

결측치(평균) / train_size : 0.9 / epochs : 2000 / batch_size : 16 / hidden_layer : 64 64 64 64 / random_state : 5772 / relu
RMSE : 40.89305346008124
loss : 1672.2418212890625
r2 : 0.7356210801180023

결측치(평균) / train_size : 0.9 / epochs : 2000 / batch_size : 16 / hidden_layer : 64 128 128 64 64 32 32 / random_state : 5772 / relu
RMSE : 37.77698325524667
loss : 1427.1004638671875
r2 : 0.7743775788920899

결측치(평균) / train_size : 0.9 / epochs : 2000 / batch_size : 16 / hidden_layer : 64 256 256 128 64 32 32 / random_state : 5772 / relu
RMSE : 38.90842739890999
loss : 1513.86572265625
r2 : 0.7606601096654704

결측치(평균) / train_size : 0.9 / epochs : 2000 / batch_size : 16 / hidden_layer : 64 512 256 128 64 64 32 / random_state : 5772 / relu
RMSE : 41.18040708558349
loss : 1695.825927734375
r2 : 0.7318924928908006

결측치(평균) / train_size : 0.9 / epochs : 2000 / batch_size : 16 / hidden_layer : 32 64 64 32 32 32 32 / random_state : 5772 / relu
RMSE : 40.58427345602142
loss : 1647.083251953125
r2 : 0.7395986361266027

결측치(평균) / train_size : 0.9 / epochs : 2000 / batch_size : 8 / hidden_layer : 64 128 128 64 64 32 32 / random_state : 5772 / relu
RMSE : 43.24005118367909
loss : 1869.7020263671875
r2 : 0.7044029359961576

결측치(평균) / train_size : 0.9 / epochs : 2000 / batch_size : 32 / hidden_layer : 64 128 128 64 64 32 32 / random_state : 5772 / relu
RMSE : 50.76241718089156
loss : 2576.822998046875
r2 : 0.5926081570564188

결측치(평균) / train_size : 0.9 / epochs : 2000 / batch_size : 12 / hidden_layer : 64 128 128 64 64 32 32 / random_state : 5772 / relu
RMSE : 34.84169171330968
loss : 1213.9434814453125
r2 : 0.8080773661369123

결측치(평균) / train_size : 0.9 / epochs : 2000 / batch_size : 10 / hidden_layer : 64 128 128 64 64 32 32 / random_state : 5772 / relu
RMSE : 35.7204433517025
loss : 1275.9500732421875
r2 : 0.7982742588697329

결측치(평균) / train_size : 0.9 / epochs : 1500 / batch_size : 12 / hidden_layer : 64 128 128 64 64 32 32 / random_state : 5772 / relu
RMSE : 38.753482391405
loss : 1501.8323974609375
r2 : 0.7625625743699544

결측치(평균) / train_size : 0.9 / epochs : 2500 / batch_size : 12 / hidden_layer : 64 128 128 64 64 32 32 / random_state : 5772 / relu
RMSE : 38.11862939858361
loss : 1453.0299072265625
r2 : 0.7702781592711414
"""
""" 결측치(평균) / train_size : 0.9 / epochs : 2000 / batch_size : 10 / hidden_layer : 64 128 128 64 64 32 32 / random_state : 5772 / relu
RMSE : 35.7204433517025     38.689255879479155
loss : 1275.9500732421875   1496.8585205078125
r2 : 0.7982742588697329     0.7633489388392657

결측치(평균) / train_size : 0.9 / epochs : 2000 / batch_size : 10 / hidden_layer : 64 128 128 64 64 32 32 / random_state : 9155 / relu
RMSE : 43.406770267760365
loss : 1884.147705078125
r2 : 0.7544079458604844

결측치(평균) / train_size : 0.9 / epochs : 2000 / batch_size : 10 / hidden_layer : 64 128 128 64 64 32 32 / random_state : 5772 / relu
RMSE : 39.00278247786464
loss : 1521.217041015625
r2 : 0.7594978679425098

결측치(평균) / train_size : 0.9 / epochs : 2000 / batch_size : 10 / hidden_layer : 64 128 128 64 64 32 32 / random_state : 6464 / relu
RMSE : 41.900849064711686
loss : 1755.68115234375
r2 : 0.7317938899370442

결측치(평균) / train_size : 0.9 / epochs : 2000 / batch_size : 10 / hidden_layer : 64 128 128 64 64 32 32 / random_state : 1686 / relu
RMSE : 48.450230181986754
loss : 2347.4248046875
r2 : 0.6542717579256727

결측치(평균) / train_size : 0.9 / epochs : 2000 / batch_size : 10 / hidden_layer : 64 128 128 64 64 32 32 / random_state : 54 / relu
RMSE : 45.414231167973384
loss : 2062.452392578125
r2 : 0.5840698120917633

결측치(평균) / train_size : 0.9 / epochs : 2000 / batch_size : 10 / hidden_layer : 64 128 128 64 64 32 32 / random_state : 5772 / relu
RMSE : 48.06455467405919
loss : 2310.201416015625
r2 : 0.634760631497936

결측치(평균) / train_size : 0.9 / epochs : 2000 / batch_size : 10 / hidden_layer : 64 128 128 64 64 32 32 / random_state : 518 / relu
RMSE : 41.07077839401909    53.45724233265779
loss : 1686.808837890625    2857.6767578125
r2 : 0.7098711571545812     0.5084834599834098

결측치(평균) / train_size : 0.9 / epochs : 2000 / batch_size : 10 / hidden_layer : 64 128 128 64 64 32 32 / random_state : 5779 / relu
RMSE : 46.2272510564318
loss : 2136.958740234375
r2 : 0.5823346898476807

결측치(평균) / train_size : 0.9 / epochs : 2000 / batch_size : 10 / hidden_layer : 64 128 128 64 64 32 32 / random_state : 5772 / relu
RMSE : 39.2079822120525     42.822665022724095
loss : 1537.265869140625    1833.7806396484375
r2 : 0.7569606066715422     0.7100820810019636

결측치(평균) / train_size : 0.9 / epochs : 2000 / batch_size : 10 / hidden_layer : 64 128 128 64 64 32 32 / random_state : 95 / relu
RMSE : 43.75503179769371
loss : 1914.5028076171875
r2 : 0.692542206129839

결측치(평균) / train_size : 0.9 / epochs : 2000 / batch_size : 10 / hidden_layer : 64 128 128 64 64 32 32 / random_state : 9371 / relu
RMSE : 44.52196586333467
loss : 1982.2054443359375
r2 : 0.7469350760098756

결측치(평균) / train_size : 0.9 / epochs : 2000 / batch_size : 10 / hidden_layer : 64 128 128 64 64 32 32 / random_state : 467 / relu
RMSE : 44.11101329759525
loss : 1945.781494140625
r2 : 0.7318144815245279

결측치(평균) / train_size : 0.9 / epochs : 2000 / batch_size : 10 / hidden_layer : 64 128 128 64 64 32 32 / random_state : 525 / relu
RMSE : 53.4749754375528
loss : 2859.572998046875
r2 : 0.5841135859636615

결측치(평균) / train_size : 0.9 / epochs : 2000 / batch_size : 10 / hidden_layer : 64 128 128 64 64 32 32 / random_state : 38 / relu


"""
""" 결측치(평균) / train_size : 0.9 / epochs : 353 / batch_size : 32 / hidden_layer : 25 50 75 50 25 / random_state : 38 / relu
RMSE : 48.432479578532835
loss : 2345.705078125
r2 : 0.6470393714074748

결측치(평균) / train_size : 0.9 / epochs : 500 / batch_size : 32 / hidden_layer : 25 50 75 50 25 / random_state : 38 / relu
RMSE : 49.81300727078596
loss : 2481.335693359375
r2 : 0.626630907685416

결측치(평균) / train_size : 0.9 / epochs : 400 / batch_size : 32 / hidden_layer : 25 50 75 50 25 / random_state : 38 / relu
RMSE : 46.69946560604976
loss : 2180.840087890625
r2 : 0.6718467325587156

결측치(평균) / train_size : 0.9 / epochs : 400 / batch_size : 16 / hidden_layer : 25 50 75 50 25 / random_state : 38 / relu
RMSE : 51.727449230618845
loss : 2675.72900390625
r2 : 0.5973803443082677

결측치(평균) / train_size : 0.9 / epochs : 400 / batch_size : 20 / hidden_layer : 25 50 75 50 25 / random_state : 38 / relu
RMSE : 49.454818296444834
loss : 2445.779052734375
r2 : 0.6319811235793588

결측치(평균) / train_size : 0.9 / epochs : 400 / batch_size : 25 / hidden_layer : 25 50 75 50 25 / random_state : 38 / relu
RMSE : 46.33510697809896
loss : 2146.942138671875
r2 : 0.6769474075386952

결측치(평균) / train_size : 0.9 / epochs : 400 / batch_size : 25 / hidden_layer : 64 64 64 64 64 / random_state : 38 / relu
RMSE : 46.370772423545525
loss : 2150.24853515625
r2 : 0.6764498899467548

결측치(평균) / train_size : 0.9 / epochs : 400 / batch_size : 25 / hidden_layer : 128 128 128 128 128 / random_state : 38 / relu
RMSE : 41.42932729548297
loss : 1716.38916015625
r2 : 0.7417331989090952

결측치(평균) / train_size : 0.9 / epochs : 400 / batch_size : 25 / hidden_layer : 256 256 256 256 256 / random_state : 38 / relu
RMSE : 45.31676973417925
loss : 2053.609619140625
r2 : 0.6909913145799498

결측치(평균) / train_size : 0.9 / epochs : 400 / batch_size : 25 / hidden_layer : 64 256 256 128 64 / random_state : 38 / relu
RMSE : 43.386020810369466
loss : 1882.3468017578125
r2 : 0.7167613727395482

결측치(평균) / train_size : 0.9 / epochs : 400 / batch_size : 25 / hidden_layer : 64 128 128 64 64 32 32 / random_state : 38 / relu
RMSE : 47.592426524605784
loss : 2265.0390625
r2 : 0.6591773021613679

결측치(평균) / train_size : 0.9 / epochs : 400 / batch_size : 25 / hidden_layer : 128 128 128 128 128 / random_state : 38 / relu
RMSE : 47.09455871900171
loss : 2217.8974609375
r2 : 0.6662706887326354

결측치(평균) / train_size : 0.9 / epochs : 400 / batch_size : 25 / hidden_layer : 64 64 64 64 64 / random_state : 38 / relu
RMSE : 47.70623403314288    47.15464243035223
loss : 2275.884765625       2223.560302734375
r2 : 0.6575452441037245     0.6654185947499145

결측치(평균) / train_size : 0.9 / epochs : 400 / batch_size : 25 / hidden_layer : 32 64 64 32 32 / random_state : 38 / relu
RMSE : 47.842851885757185
loss : 2288.9384765625
r2 : 0.6555811167375629

결측치(평균) / train_size : 0.9 / epochs : 400 / batch_size : 25 / hidden_layer : 64 128 128 64 32 / random_state : 38 / relu
RMSE : 47.2331138187567
loss : 2230.967041015625
r2 : 0.6643040987705898

결측치(평균) / train_size : 0.9 / epochs : 400 / batch_size : 25 / hidden_layer : 64 128 128 64 32 32 / random_state : 38 / relu
RMSE : 44.17436072551376
loss : 1951.3741455078125
r2 : 0.7063747572089241

결측치(평균) / train_size : 0.9 / epochs : 400 / batch_size : 25 / hidden_layer : 64 128 128 64 64 32 32 / random_state : 38 / relu
RMSE : 48.610236351256304
loss : 2362.955078125
r2 : 0.6444437703107228

결측치(평균) / train_size : 0.9 / epochs : 400 / batch_size : 25 / hidden_layer : 64 128 128 64 32 32 / random_state : 38 / relu
RMSE : 44.17436072551376    46.850363317361264  45.76271755872213
loss : 1951.3741455078125   2194.95654296875    2094.226318359375
r2 : 0.7063747572089241     0.66972267289505    0.6848796287435537

결측치(평균) / train_size : 0.9 / epochs : 400 / batch_size : 25 / hidden_layer : 32 64 64 32 16 16 / random_state : 38 / relu
RMSE : 46.86552768877008    47.55139591963889
loss : 2196.377685546875    2261.13525390625
r2 : 0.6695088329405194     0.6597646870669776
"""
""" 결측치(평균) / train_size : 0.9 / epochs : 1000 / batch_size : 25 / hidden_layer : 64 128 128 64 32 32 / random_state : 38 / relu
RMSE : 45.93012270694473
loss : 2109.576171875
r2 : 0.682569897535326

결측치(평균) / train_size : 0.9 / epochs : 2000 / batch_size : 25 / hidden_layer : 64 128 128 64 32 32 / random_state : 38 / relu
RMSE : 46.07654339859382
loss : 2123.0478515625
r2 : 0.6805428555256808

결측치(평균) / train_size : 0.9 / epochs : 1500 / batch_size : 25 / hidden_layer : 64 128 128 64 32 32 / random_state : 38 / relu
RMSE : 48.675521532741435
loss : 2369.306396484375
r2 : 0.6434880698021239

결측치(평균) / train_size : 0.9 / epochs : 700 / batch_size : 25 / hidden_layer : 64 128 128 64 32 32 / random_state : 38 / relu
RMSE : 40.78673933362104
loss : 1663.55810546875
r2 : 0.7496827083029085

결측치(평균) / train_size : 0.9 / epochs : 550 / batch_size : 25 / hidden_layer : 64 128 128 64 32 32 / random_state : 38 / relu
RMSE : 52.84699548357148
loss : 2792.804931640625
r2 : 0.5797637937289575

결측치(평균) / train_size : 0.9 / epochs : 850 / batch_size : 25 / hidden_layer : 64 128 128 64 32 32 / random_state : 38 / relu
RMSE : 43.93227376012281
loss : 1930.044677734375
r2 : 0.7095841904791147

결측치(평균) / train_size : 0.9 / epochs : 700 / batch_size : 25 / hidden_layer : 64 128 128 64 32 32 / random_state : 38 / relu
RMSE : 40.78673933362104    47.88378315727701   45.3032587559245    41.76949161394594   48.53985908428119
loss : 1663.55810546875     2292.856689453125   2052.38525390625    1744.6904296875     2356.117919921875
r2 : 0.7496827083029085     0.6549915350244913  0.6911754705850939  0.7374746492546277  0.6454725507612229

결측치(평균) / train_size : 0.9 / epochs : 700 / batch_size : 20 / hidden_layer : 64 128 128 64 32 32 / random_state : 38 / relu
RMSE : 40.84476615032825    43.063364582213325  42.87845199523663   43.70282193986805   45.11731534072797
loss : 1668.294921875       1854.453369140625   1838.5616455078125  1909.9366455078125  2035.5721435546875
r2 : 0.74896995016778       0.7209585209062517  0.7233497621178264  0.7126098909772234  0.6937053834382036

결측치(평균) / train_size : 0.9 / epochs : 700 / batch_size : 20 / hidden_layer : 64 128 128 64 32 32 / random_state : 38 / relu
RMSE : 44.22875052549953    57.33972014109744   44.4322583933959
loss : 1956.182373046875    3287.843505859375   1974.2255859375
r2 : 0.705651256244894      0.5052748507538525  0.7029362556256006
"""
"""결측치(평균) / train_size : 0.9 / epochs : 300 / batch_size : 32 / hidden_layer : 64 128 128 64 32 32 / random_state : 38 / relu
RMSE : 48.77348442753629    47.488845318532434  
loss : 2378.852783203125    2255.1904296875     
r2 : 0.642051580748411      0.6606591923664498  

결측치(평균) / train_size : 0.9 / epochs : 300 / batch_size : 32 / hidden_layer : 64 128 128 64 32 / random_state : 38 / relu
RMSE : 43.720084638312976   
loss : 1911.44580078125     
r2 : 0.7123828044256969

결측치(평균) / train_size : 0.9 / epochs : 300 / batch_size : 32 / hidden_layer : 64 128 128 64 32 / random_state : 976 / relu
RMSE : 55.647506858180535
loss : 3096.64501953125
r2 : 0.5447985447554902

결측치(평균) / train_size : 0.9 / epochs : 300 / batch_size : 32 / hidden_layer : 64 128 128 64 32 / random_state : 535 / relu
RMSE : 48.34803986703468
loss : 2337.532958984375
r2 : 0.6508250283384067

결측치(평균) / train_size : 0.9 / epochs : 300 / batch_size : 32 / hidden_layer : 64 128 128 64 32 / random_state : 68 / relu
RMSE : 50.18439874057599
loss : 2518.473876953125
r2 : 0.6045287349414452

결측치(평균) / train_size : 0.9 / epochs : 300 / batch_size : 32 / hidden_layer : 64 128 128 64 32 / random_state : 7599 / relu
RMSE : 51.140243085955554
loss : 2615.324462890625
r2 : 0.6218715808042863

결측치(평균) / train_size : 0.9 / epochs : 300 / batch_size : 16 / hidden_layer : 64 128 128 64 32 / random_state : 7599 / relu
RMSE : 48.221425820926584
loss : 2325.305908203125
r2 : 0.6638030140044816

결측치(평균) / train_size : 0.9 / epochs : 300 / batch_size : 32 / hidden_layer : 64 128 128 64 32 / random_state : 84 / relu
RMSE : 53.22532347101096
loss : 2832.93505859375
r2 : 0.5941498703005459

결측치(평균) / train_size : 0.9 / epochs : 300 / batch_size : 32 / hidden_layer : 64 128 128 64 32 / random_state : 38 / relu
RMSE : 50.39663481603955
loss : 2539.82080078125
r2 : 0.6178305533070789

결측치(평균) / train_size : 0.9 / epochs : 500 / batch_size : 32 / hidden_layer : 64 128 128 64 32 / random_state : 38 / relu
RMSE : 39.842489257014456
loss : 1587.4239501953125
r2 : 0.7611386800709575

결측치(평균) / train_size : 0.9 / epochs : 300 / batch_size : 32 / hidden_layer : 64 128 128 64 32 / random_state : 38 / relu

""" 

# submission.csv에 test_csv의 예측값 넣기
y_submit = model.predict(test_csv)
    # train 데이터의 shape와 동일한 컬럼을 확인하고 넣어야함
    # x_train.shape:(N, 9)
print(y_submit.shape) # (715, 1)

### submission.csv 파일 만들기 
submission_csv['count'] = y_submit
print(submission_csv)

submission_csv.to_csv(path + 'submission_0522_1050.csv')
    # 명령어의 내용을 해당 경로로 파일 생성