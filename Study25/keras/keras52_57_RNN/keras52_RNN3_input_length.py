# 52-1.copy
 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU

import numpy as np

##########################################################################
#1. 데이터
##########################################################################

datasets = np.array([1,2,3,4,5,6,7,8,9,10])     
# 데이터 의미를 부여하기 나름 시계열 데이터로 해석 할 수 있다.
# 실제로 시계열 데이터는 2차원으로 제공될 수도 있음
# 또한, y값도 제공되지 않고, 우리가 직접 부여해야한다.
# 이 때, 우리는 Time Step에 따라서 잘라 줘야한다.            
      
x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9]])          
# datasets을 Time Step = 3 으로 자른 경우
# 통상적으로 Time Step이 크면 모델 성능이 증가하지만 최적의 값은 몇인지 알아야한다
y = np.array([4,5,6,7,8,9,10])

""" 데이터 해석
x[0] 다음에는 y[0],
x[1] 다음에는 y[1],
x[2] 다음에는 y[2],
x[3] 다음에는 y[3],
x[4] 다음에는 y[4]가 나올것이다
"""
# print(x.shape)  (7, 3)    : (N-TS, TS) >> 3차원 데이터가 아니다? feature가 빠짐
# print(y.shape)  (7,)      : (N-TS, )

x = x.reshape(x.shape[0], x.shape[1], 1)
# print(x.shape)  (7, 3, 1) : (N-TS, TS, F)
#                             (batch_size, time_step, feature)
""" print(x)
x = np.array([[[1],[2],[3]],
              [[2],[3],[4]],
              [[3],[4],[5]],
              [[4],[5],[6]],
              [[5],[6],[7]],
              [[6],[7],[8]],
              [[7],[8],[9]]])
"""
# time_step 만큼 잘라진 데이터는 모델 훈련을 할 때, feature 단위로 훈련 진행

##########################################################################
#2. 모델 구성
##########################################################################
model = Sequential()
model.add(SimpleRNN(units=100, input_length=3, input_dim=1))
                                               # input_shape = (time step, feature)
                                               # input_length = time step
                                               # input_dim = feature
# model.add(SimpleRNN(units=130, input_shape = (3,1), activation='relu'))
                                               # parameter 이름을 명시했을 때는 순서가 바뀌어도 무관
model.add(Dense(100, activation='relu'))       # Simple RNN의 경우 3차원 데이터가 들어가면 2차원이 나옴
model.add(Dense(100, activation='relu'))       # : Time Step 이후에 단일 시간의 결과값 > 2차원
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))

##########################################################################
#3. 컴파일 훈련
##########################################################################
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y,
          epochs = 1000)

##########################################################################
#4. 평가 예측
##########################################################################
loss = model.evaluate(x, y)

x_prd = np.array([8,9,10]).reshape(1,3,1)
y_prd = model.predict(x_prd)

print('lss :', loss)
print('rst :', y_prd)

''' [SimpleRNN 기준]
[ SimpleRNN ]
lss : 1.4025515326920868e-07
rst : [[11.001318]]
[ LSTM ] : 시계열에서는 최강 but 뒤지게 느리다
lss : 1.8311702660867013e-05
rst : [[10.992816]]
[ GRU ]
lss : 0.0007120014051906765
rst : [[10.907407]]
'''

''' [LSTM 기준]
[ SimpleRNN ]
lss : 2.9083145136610256e-07
rst : [[11.00916]]
[ LSTM ]
lss : 2.6687863282859325e-05
rst : [[11.000248]]
[ GRU ]
lss : 0.00039025116711854935
rst : [[10.922967]]
'''

''' [GRU 기준]
[ SimpleRNN ]
lss : 5.970498762053467e-08
rst : [[11.047629]]
[ LSTM ]
lss : 2.146683618775569e-05
rst : [[11.007244]]
[ GRU ]
lss : 3.3151347906823503e-06
rst : [[10.992953]]
'''
