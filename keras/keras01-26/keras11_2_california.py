# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
    # 오류나서 chat GTP로 해결한 코드

import sklearn as sk
print(sk.__version__)   # 1.1.3
import tensorflow as tf
print(tf.__version__)   # 2.9.3

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import numpy as np

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x)
print(y)        # 데이터를 찍어보고 소수점(회귀) or 정수 몇개로만 구성(분류) AI 모델 종류 결정
print(x.shape)  # (20640, 8)
print(y.shape)  # (20640,)
print(datasets.feature_names)
                # ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
x_trn, x_tst, y_trn, y_tst = train_test_split(x, y,
                                              test_size= 0.15,
                                              shuffle= True,
                                              random_state= 55)


#2. 모델구성
model = Sequential()
model.add(Dense(64,input_dim=8))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(1))
epochs = 7500

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_trn, y_trn, epochs = epochs, batch_size = 50)

#4. 평가, 예측
loss = model.evaluate(x_tst, y_tst)
results = model.predict([x_tst])

from sklearn.metrics import r2_score, mean_squared_error
def RMSE(a, b) :
    return np.sqrt(mean_squared_error(a,b))

rmse = RMSE(y_tst, results)
R2 = r2_score(y_tst, results)

print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print('loss :', loss)
print('rmse :', rmse)
print('R2 :', R2)



# trn=0.85 / RS= 55 / e = 500 / HL 8-64-64-64-64-1 / BS=50
# loss : 0.6304769515991211
# rmse : 0.7940257859485748
# R2 : 0.5290073904403279 ***

# trn=0.85 / RS= 55 / e = 500 / HL 8-64-64-64-64-64-1 / BS=100          # trn=0.85 / RS= 55 / e = 500 / HL 8-128-128-64-64-32-1 / BS=50
# loss : 0.6456552743911743                                             # loss : 0.6435232162475586
# rmse : 0.8035267300743124                                             # rmse : 0.8021989756428142
# R2 : 0.5176685984703748                                               # R2 : 0.5192612985413991

# trn=0.85 / RS= 55 / e = 500 / HL 8-64-64-64-64-64-1 / BS=75
# loss : 0.6357957124710083
# rmse : 0.797368014979629
# R2 : 0.525034022812785 ***

# trn=0.85 / RS= 55 / e = 500 / HL 8-64-128-256-128-64-1 / BS=50        # trn=0.85 / RS= 55 / e = 500 / HL 8-64-128-128-64-64-1 / BS=50
# loss : 0.6507092714309692                                             # loss : 0.6299945712089539
# rmse : 0.8066655720795856                                             # rmse : 0.7937219941532706
# R2 : 0.513892945400295                                                # R2 : 0.5293677221103499 ***

# trn=0.85 / RS= 55 / e = 500 / HL 8-64-128-128-64-32-16-1 / BS=50      # trn=0.85 / RS= 55 / e = 500 / HL 8-64-128-128-64-64-1 / BS=70
# loss : 0.6446413993835449                                             # loss : 0.6346902251243591
# rmse : 0.8028956186560392                                             # rmse : 0.7966744698923017
# R2 : 0.5184259729288581                                               # R2 : 0.5258599076101458 ***

# trn=0.85 / RS= 55 / e = 500 / HL 8-64-128-128-64-64-1 / BS=40         # trn=0.85 / RS= 55 / e = 400 / HL 8-64-128-128-64-64-1 / BS=70
# loss : 0.6488192677497864                                             # loss : 0.6557721495628357
# rmse : 0.8054931690999824                                             # rmse : 0.809797612814973
# R2 : 0.5153049288159129                                               # R2 : 0.5101108010544397

# trn=0.85 / RS= 55 / e = 500 / HL 8-64-128-128-64-64-1 / BS=100        # trn=0.85 / RS= 55 / e = 1000 / HL 8-64-128-128-64-64-1 / BS=200
# loss : 0.6390752196311951                                             # loss : 0.6401686072349548
# rmse : 0.7994217858748356                                             # rmse : 0.8001053711505709
# R2 : 0.5225841438698902                                               # R2 : 0.5217673185388645

# trn=0.85 / RS= 55 / e = 1500 / HL 8-64-128-128-64-64-1 / BS=100       # trn=0.85 / RS= 55 / e = 2000 / HL 8-64-128-128-64-64-1 / BS=100
# loss : 0.6198385953903198                                             # loss : 0.5991955995559692
# rmse : 0.7872983234977254                                             # rmse : 0.7740772635570702
# R2 : 0.5369546437287722                                               # R2 : 0.5523758568352948 ********

# trn=0.85 / RS= 55 / e = 3000 / HL 8-64-128-128-64-64-1 / BS=50        # trn=0.85 / RS= 55 / e = 5000 / HL 8-64-128-128-64-64-1 / BS=50
# loss : 0.5708311796188354                                             # loss : 0.5563498139381409
# rmse : 0.7555336430765923                                             # rmse : 0.7458885803295967
# R2 : 0.5735653408726058 ********                                      # R2 : 0.5843834853462511 ***********

# trn=0.85 / RS= 55 / e = 5000 / HL 8-64-128-128-64-64-1 / BS=100       # trn=0.85 / RS= 55 / e = 7500 / HL 8-64-128-128-64-64-1 / BS=50
# loss : 0.5921935439109802                                             # loss : 0.5569707155227661
# rmse : 0.7695411282450344                                             # rmse : 0.7463047158384489
# R2 : 0.557606689478624                                                # R2 : 0.5839196063164579