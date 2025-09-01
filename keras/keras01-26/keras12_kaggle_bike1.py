# https://www.kaggle.com/competitions/bike-sharing-demand/overview
# 지금은 시간은 data XX / 나중에 정제하면 연 월 일 시 등으로 사용가능

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터
path = './_data/kaggle/bike/'           # 상대 경로
''' 경로 지정 '/' 사용 tip
path = '.\_data\kaggle\bike\'           # \ 도 가능은 하지만 \n, a, b 같이 선 예약된 애들이 있으면 중간에 먼저 코드로 인식
path = '.\\_data\\kaggle\\bike\\'       # //, \\ 는 그냥 가능   
path = './/_data//kaggle//bike//'       
path = 'c:/Study25/_data/kaggle/bike/'  # 절대 경로
'''

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
samplesubmission_csv = pd.read_csv(path + 'samplesubmission.csv')

'''print(train_csv)
                   datetime  season  holiday  workingday  weather   temp   atemp  humidity  windspeed  casual  registered  count
2011-01-01 00:00:00       1        0           0        1   9.84  14.395        81     0.0000       3          13     16
2011-01-01 01:00:00       1        0           0        1   9.02  13.635        80     0.0000       8          32     40
2011-01-01 02:00:00       1        0           0        1   9.02  13.635        80     0.0000       5          27     32
2011-01-01 03:00:00       1        0           0        1   9.84  14.395        75     0.0000       3          10     13
2011-01-01 04:00:00       1        0           0        1   9.84  14.395        75     0.0000       0           1      1
                ...     ...      ...         ...      ...    ...     ...       ...        ...     ...         ...    ...
2012-12-19 19:00:00       4        0           1        1  15.58  19.695        50    26.0027       7         329    336
2012-12-19 20:00:00       4        0           1        1  14.76  17.425        57    15.0013      10         231    241
2012-12-19 21:00:00       4        0           1        1  13.94  15.910        61    15.0013       4         164    168
2012-12-19 22:00:00       4        0           1        1  13.94  17.425        61     6.0032      12         117    129
2012-12-19 23:00:00       4        0           1        1  13.12  16.665        66     8.9981       4          84     88
'''
## print(train_csv.shape) (10886, 11)
## print(train_csv.columns) ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count']
'''print(train_csv.info())
 #   Column      Non-Null Count  Dtype
---  ------      --------------  -----
 0   season      10886 non-null  int64
 1   holiday     10886 non-null  int64
 2   workingday  10886 non-null  int64
 3   weather     10886 non-null  int64
 4   temp        10886 non-null  float64
 5   atemp       10886 non-null  float64
 6   humidity    10886 non-null  int64
 7   windspeed   10886 non-null  float64
 8   casual      10886 non-null  int64
 9   registered  10886 non-null  int64
 10  count       10886 non-null  int64
'''
## print(train_csv.isnull().sum())  없음
## print(train_csv.isna().sum())    없음
'''print(train_csv.describe())
             season       holiday    workingday       weather         temp         atemp      humidity     windspeed        casual    registered         count
count  10886.000000  10886.000000  10886.000000  10886.000000  10886.00000  10886.000000  10886.000000  10886.000000  10886.000000  10886.000000  10886.000000
mean       2.506614      0.028569      0.680875      1.418427     20.23086     23.655084     61.886460     12.799395     36.021955    155.552177    191.574132
std        1.116174      0.166599      0.466159      0.633839      7.79159      8.474601     19.245033      8.164537     49.960477    151.039033    181.144454
min        1.000000      0.000000      0.000000      1.000000      0.82000      0.760000      0.000000      0.000000      0.000000      0.000000      1.000000
25%        2.000000      0.000000      0.000000      1.000000     13.94000     16.665000     47.000000      7.001500      4.000000     36.000000     42.000000
50%        3.000000      0.000000      1.000000      1.000000     20.50000     24.240000     62.000000     12.998000     17.000000    118.000000    145.000000
75%        4.000000      0.000000      1.000000      2.000000     26.24000     31.060000     77.000000     16.997900     49.000000    222.000000    284.000000
max        4.000000      1.000000      1.000000      4.000000     41.00000     45.455000    100.000000     56.996900    367.000000    886.000000    977.000000
season : 봄 여름 가을 겨울 > 1, 2, 3, 4
holiday : 평일 휴일 > 0, 1 등등으로 분석...! > 나중에 이상한 데이터도 잡아내야함
'''

'''print(test_csv)
           datetime  season  holiday  workingday  weather   temp   atemp  humidity  windspeed
2011-01-20 00:00:00       1        0           1        1  10.66  11.365        56    26.0027
2011-01-20 01:00:00       1        0           1        1  10.66  13.635        56     0.0000
2011-01-20 02:00:00       1        0           1        1  10.66  13.635        56     0.0000
2011-01-20 03:00:00       1        0           1        1  10.66  12.880        56    11.0014
2011-01-20 04:00:00       1        0           1        1  10.66  12.880        56    11.0014
                ...     ...      ...         ...      ...    ...     ...       ...        ...
2012-12-31 19:00:00       1        0           1        2  10.66  12.880        60    11.0014
2012-12-31 20:00:00       1        0           1        2  10.66  12.880        60    11.0014
2012-12-31 21:00:00       1        0           1        1  10.66  12.880        60    11.0014
2012-12-31 22:00:00       1        0           1        1  10.66  13.635        56     8.9981
2012-12-31 23:00:00       1        0           1        1  10.66  13.635        65     8.9981
'''
## print(test_csv.columns)  ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed']
## print(test_csv.shape)  (6493, 8)
'''print(test_csv.info()) =isna(), =isnull()
 #   Column      Non-Null Count  Dtype
---  ------      --------------  -----
 0   datetime    6493 non-null   object
 1   season      6493 non-null   int64
 2   holiday     6493 non-null   int64
 3   workingday  6493 non-null   int64
 4   weather     6493 non-null   int64
 5   temp        6493 non-null   float64
 6   atemp       6493 non-null   float64
 7   humidity    6493 non-null   int64
 8   windspeed   6493 non-null   float64
'''

'''print(samplesubmission_scv)
                 datetime  count
0     2011-01-20 00:00:00      0
1     2011-01-20 01:00:00      0
2     2011-01-20 02:00:00      0
3     2011-01-20 03:00:00      0
4     2011-01-20 04:00:00      0
...                   ...    ...
6488  2012-12-31 19:00:00      0
6489  2012-12-31 20:00:00      0
6490  2012-12-31 21:00:00      0
6491  2012-12-31 22:00:00      0
6492  2012-12-31 23:00:00      0
'''
## print(samplesubmission_scv.shape)    (6493, 1)
## print(samplesubmission_scv.columns)  ['count']

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
'''print(x)
                     season  holiday  workingday  weather   temp   atemp  humidity  windspeed
datetime
2011-01-01 00:00:00       1        0           0        1   9.84  14.395        81     0.0000
2011-01-01 01:00:00       1        0           0        1   9.02  13.635        80     0.0000
2011-01-01 02:00:00       1        0           0        1   9.02  13.635        80     0.0000
2011-01-01 03:00:00       1        0           0        1   9.84  14.395        75     0.0000
2011-01-01 04:00:00       1        0           0        1   9.84  14.395        75     0.0000
...                     ...      ...         ...      ...    ...     ...       ...        ...
2012-12-19 19:00:00       4        0           1        1  15.58  19.695        50    26.0027
2012-12-19 20:00:00       4        0           1        1  14.76  17.425        57    15.0013
2012-12-19 21:00:00       4        0           1        1  13.94  15.910        61    15.0013
2012-12-19 22:00:00       4        0           1        1  13.94  17.425        61     6.0032
2012-12-19 23:00:00       4        0           1        1  13.12  16.665        66     8.9981
'''
## print(x.shape) (10886, 8)

## 저 이상한 column은 뭘까?? : column engineering과 파생 feature
 # column 끼리 조합해서 새로운 열을 만들거나 하나의 feature에서 여러게의 열을 추가시킬 수도 있다.
 # 이러한 작업은 모델의 완성도를 올리는데 크게 기여!!
 # keggle bike의 경우 train에는 casual, registered, count가 있는데, test에는 없음 : 우리가 구할 것은 count
 # >> 활용방안 >> (n,8)-(n,2) 모델로 reg, casual  먼저 찾고, (n,10)-(n,1) 모델로 count 추적

y = train_csv['count']
'''print(y)
datetime
2011-01-01 00:00:00     16
2011-01-01 01:00:00     40
2011-01-01 02:00:00     32
2011-01-01 03:00:00     13
2011-01-01 04:00:00      1
                      ...
2012-12-19 19:00:00    336
2012-12-19 20:00:00    241
2012-12-19 21:00:00    168
2012-12-19 22:00:00    129
2012-12-19 23:00:00     88
'''
## print(y.shape) (10886,)

x_trn, x_tst, y_trn, y_tst = train_test_split(x, y,
                                              test_size=0.2,
                                              random_state=3081)

#2. 모델구성
model = Sequential()
model.add(Dense(50, input_dim=8, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(50, activation= 'relu'))
model.add(Dense(50, activation= 'relu'))
model.add(Dense(1, activation= 'linear'))
                            # Default : activation= linear // 생락가능

epochs = 200

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_trn, y_trn, epochs = epochs, batch_size=14)

#4. 평가, 예측
results = model.predict(x_tst)
def RMSE(a,b):
    return np.sqrt(mean_squared_error(a,b))
rmse = RMSE(y_tst, results)
R2 = r2_score(y_tst, results)

print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print('rmse :', rmse)
print('R2   :', R2)

''' Submitted
rmse : 145.16873504687112   145.32943960837358      145.84739525437564      145.02188692543407      144.90467792851825
R2   : 0.3397677275341291   0.3357725721241318      0.3310295079641524      0.33858093317759563     0.3396496399615667
score: 1.28850              1.31614                 1.331313                1.28570 ****            1.31725

rmse : 145.08176705244185
R2   : 0.3380346151662057
score: 
'''

## Tuning
''' train_size : 0.8 / epochs : 200 / batch_size : 16 / hidden_layer : 64 64 64 64 64 / random_state : 42 / activation : relu
train_size : 0.8 / epochs : 100 / batch_size : 32 / hidden_layer : 64 64 64 64 64 / random_state : 42 / activation : relu
rmse : 149.24718172053036   149.34105752479894      149.59462905570717
R2   : 0.32515028530363854  0.32430106345466403     0.32200452856880646

train_size : 0.8 / epochs : 200 / batch_size : 32 / hidden_layer : 64 64 64 64 64 / random_state : 42 / activation : relu
rmse : 147.13136639600629   149.22740242654567      150.19855627749232
R2   : 0.3441487853925068   0.325329145182391       0.3165192188669692

train_size : 0.8 / epochs : 300 / batch_size : 32 / hidden_layer : 64 64 64 64 64 / random_state : 42 / activation : relu
rmse : 150.1140525358047    150.75370844981063      148.7359087256794
R2   : 0.3172880736078173   0.3114574250606422      0.32976600327547123

train_size : 0.8 / epochs : 200 / batch_size : 16 / hidden_layer : 64 64 64 64 64 / random_state : 42 / activation : relu
rmse : 147.34321688571927   149.00286290094155      148.40033763866242
R2   : 0.34225874067792483  0.3273579455100324      0.3327868935996243

train_size : 0.8 / epochs : 200 / batch_size : 8 / hidden_layer : 64 64 64 64 64 / random_state : 42 / activation : relu
rmse : 150.46708448057498   
R2   : 0.3140731510292398

train_size : 0.8 / epochs : 200 / batch_size : 16 / hidden_layer : 64 64 64 64 64 / random_state : 17 / activation : relu
rmse : 153.91478321672435
R2   : 0.29764277485183976

train_size : 0.8 / epochs : 200 / batch_size : 25 / hidden_layer : 64 64 64 64 64 / random_state : 42 / activation : relu
rmse : 148.42709768779878
R2   : 0.3325462436711718

train_size : 0.8 / epochs : 200 / batch_size : 16 / hidden_layer : 64 64 64 64 64 / random_state : 42 / activation : relu
rmse : 148.08564830752553   149.32515596088894
R2   : 0.33561360180999467  0.3244449501789852
'''
''' train_size : 0.8 / epochs : 200 / batch_size : 16 / hidden_layer : 64 64 64 / random_state : 42 / activation : relu
train_size : 0.8 / epochs : 200 / batch_size : 16 / hidden_layer : 64 64 64 64 64 / random_state : 42 / activation : relu
rmse : 148.08564830752553   149.32515596088894  147.34321688571927   149.00286290094155      148.40033763866242
R2   : 0.33561360180999467  0.3244449501789852  0.34225874067792483  0.3273579455100324      0.3327868935996243

train_size : 0.8 / epochs : 200 / batch_size : 16 / hidden_layer : 64 64 64 64 / random_state : 42 / activation : relu
rmse : 148.30845734603096   148.37133957566698  149.969731497778    147.92928031081527      150.44038637685156
R2   : 0.333612831851542    0.3330476207336518  0.3186001736883515  0.3370159513517288      0.3143165440815655

train_size : 0.8 / epochs : 200 / batch_size : 16 / hidden_layer : 64 64 64 / random_state : 42 / activation : relu
rmse : 146.65411499667158   146.65411499667158  146.32880353359812  147.4371140031306       147.53430049044928
R2   : 0.3483966662825171   0.3483966662825171  0.3512842624614635  0.3414201586461604      0.3405516370799273

train_size : 0.8 / epochs : 200 / batch_size : 16 / hidden_layer : 64 64 / random_state : 42 / activation : relu
rmse : 147.51693868126787   147.08342832756958  147.46713525678203  147.0960675506171       147.25956646641728
R2   : 0.34070683547017433  0.34457609221741836 0.34115193031663227 0.34446344315845656     0.34300536091724965
'''
''' train_size : 0.8 / epochs : 200 / batch_size : 16 / hidden_layer : 64 64 64 / random_state : 42 / activation : relu
train_size : 0.8 / epochs : 200 / batch_size : 16 / hidden_layer : 64 64 64 / random_state : 42 / activation : relu
rmse : 147.69011987668455   147.11222611893632  149.21286746218522  147.83797053760173      147.74349975876706
R2   : 0.33915793945917994  0.3443194132844697  0.32546056661164124 0.3378341563711037      0.3386801546734659

train_size : 0.8 / epochs : 200 / batch_size : 16 / hidden_layer : 64 64 64 / random_state : 4245 / activation : relu
rmse : 145.01432619511053
R2   : 0.3411714938441911
'''
## 레이어가 적으면 가중치가 낮아서 적절하지 않은 값이 나올수도...! 
 # ex) 결과가 횟수인데 보는데 음수
''' train_size : 0.8 / epochs : 200 / batch_size : 16 / hidden_layer : 64 64 64 64 64 / random_state : 4245 / activation : relu
train_size : 0.8 / epochs : 200 / batch_size : 16 / hidden_layer : 64 64 64 64 64 / random_state : 42 / activation : relu
rmse : 148.08564830752553   149.32515596088894  147.34321688571927   149.00286290094155      148.40033763866242
R2   : 0.33561360180999467  0.3244449501789852  0.34225874067792483  0.3273579455100324      0.3327868935996243

train_size : 0.8 / epochs : 200 / batch_size : 16 / hidden_layer : 64 64 64 64 64 / random_state : 4245 / activation : relu
rmse : 145.16873504687112   148.16651529003204  148.13393445721923   146.1513961504457      147.76303884853922
R2   : 0.3397677275341291   0.312218170984077   0.31252061504167394  0.3307991239874092     0.31595890721134956

train_size : 0.8 / epochs : 200 / batch_size : 16 / hidden_layer : 64 64 64 64 64 64 / random_state : 4245 / activation : relu
rmse : 148.8513650125301    146.87267319057034  149.41204263545916   146.8328119751145      147.19076696000764
R2   : 0.3058453980811937   0.32417763033496383 0.3006062127350282   0.3245444166794692     0.3212470963639966

train_size : 0.8 / epochs : 200 / batch_size : 16 / hidden_layer : 64 64 64 64 64 64 64 / random_state : 4245 / activation : relu
rmse : 148.25406221079567   148.4596844559423
R2   : 0.31140515368115285  0.30949372398800223
'''
''' train_size : 0.8 / epochs : 200 / batch_size : 16 / hidden_layer : 64 128 64 64 32 / random_state : 4245 / activation : relu
train_size : 0.8 / epochs : 200 / batch_size : 16 / hidden_layer : 64 64 64 64 64 / random_state : 4245 / activation : relu
rmse : 145.16873504687112   148.16651529003204  148.13393445721923   146.1513961504457      147.76303884853922
R2   : 0.3397677275341291   0.312218170984077   0.31252061504167394  0.3307991239874092     0.31595890721134956

train_size : 0.8 / epochs : 200 / batch_size : 16 / hidden_layer : 64 128 128 64 46 / random_state : 4245 / activation : relu
rmse : 147.3160770108443    149.06928571243327  146.91948940466816   146.87925374528848     148.15958379235056
R2   : 0.3200908993010403   0.3038114041306704  0.32374671985095604  0.324117069229676      0.3122825208357427

train_size : 0.8 / epochs : 200 / batch_size : 16 / hidden_layer : 64 128 64 64 32 / random_state : 4245 / activation : relu
rmse : 147.43807928056495   146.24405373471768  150.079366272752     145.60616325438983
R2   : 0.31896427682269857  0.3299503302877177  0.29434481284720126  0.33578285658269413
'''
## count가 음수!!! 음수면 레이어랑 노드를 늘려서 가중치 조절해서 해결 + 
''' train_size : 0.8 / epochs : 200 / batch_size : 14 / hidden_layer : 64 64 64 64 64 32 / random_state : 3081 / activation : relu
train_size : 0.8 / epochs : 200 / batch_size : 16 / hidden_layer : 64 64 64 64 64 32 / random_state : 3081 / activation : relu
rmse : 145.32943960837358   146.4343552470358   145.70987723228     146.75018745788944
R2   : 0.3357725721241318   0.32563415567177767 0.33229044413563114 0.32272205022140876

train_size : 0.8 / epochs : 300 / batch_size : 16 / hidden_layer : 64 64 64 64 64 32 / random_state : 3081 / activation : relu
rmse : 148.89938132628316
R2   : 0.30273896917120324

train_size : 0.8 / epochs : 100 / batch_size : 16 / hidden_layer : 64 64 64 64 64 32 / random_state : 3081 / activation : relu
rmse : 150.2971425298111
R2   : 0.2895867471200859

train_size : 0.8 / epochs : 250 / batch_size : 16 / hidden_layer : 64 64 64 64 64 32 / random_state : 3081 / activation : relu
rmse : 147.09430518381674
R2   : 0.3195419983303295

train_size : 0.8 / epochs : 225 / batch_size : 16 / hidden_layer : 64 64 64 64 64 32 / random_state : 3081 / activation : relu
rmse : 146.11902052018388   148.98299971025293  147.2383597718931
R2   : 0.32853541481269677  0.3019556185472443  0.31820855315402985

train_size : 0.8 / epochs : 200 / batch_size : 12 / hidden_layer : 64 64 64 64 64 32 / random_state : 3081 / activation : relu
rmse : 145.84739525437564   146.98988085080973      145.02188692543407
R2   : 0.3310295079641524   0.32050778896255716     0.33858093317759563
'''
'''
train_size : 0.8 / epochs : 200 / batch_size : 14 / hidden_layer : 64 64 64 64 64 32 / random_state : 3081 / activation : relu
rmse : 146.92478534956425   147.01721614138364      144.90467792851825
R2   : 0.32110949149627177  0.320255038990766       0.3396496399615667

train_size : 0.8 / epochs : 200 / batch_size : 14 / hidden_layer : 50 100 100 100 50 50 / random_state : 3081 / activation : relu
rmse : 147.3653517923303    145.08176705244185
R2   : 0.31703196558959323  0.3380346151662057
'''

## 파일 제출
y_submit = model.predict(test_csv)
samplesubmission_csv['count'] = y_submit
samplesubmission_csv.to_csv(path + 'samplesubmission_0522_2.csv', index=False)

### 제출 파일 만들어서 내는 사람들은 거절 당한다고 함
  # Why? index도 column으로 형성됨
  #     > index=False 하면 index 없이 생성됨!!
  #     > Default : index = True