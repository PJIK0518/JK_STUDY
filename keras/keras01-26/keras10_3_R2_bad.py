# 20250611 [보강]

#1. R2를 음수가 아닌 0.5 이하로 만들것
#2. 데이터 건들지말고
#3. Layer는 7개 이상
#4. batch_size = 1
#5. 히든레이어 노드 10~100
#6. train size = 0.75
#7. epochs 100 이상
#8. loss 지표는 mse
#9. dropout 금지


from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
# y = np.array([1,2,4,3,5,7,9,3,8,12,13, 8,14,15, 9, 6,17,23,21,20])
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size = 0.75,
                                                    shuffle=False,
                                                    random_state = 333)

#2. 모델구성
path = './Study25/_save/'
model = load_model(path + 'save.h5')
# model = Sequential()
# model.add(Dense(10, input_dim = 1))
# model.add(Dense(100, activation='sigmoid'))
# model.add(Dense(100, activation='sigmoid'))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(1))

# epochs = 100


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
# model.fit(x_train, y_train, epochs = 100, batch_size=1)
# model.save(path + 'save.h5')

#4. 평가, 예측
l = model.evaluate(x_test, y_test)
r = model.predict(x_test)

print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print('loss :', l)
print('[x]의 결과값 :', r)

from sklearn.metrics import r2_score, mean_squared_error

# def RMSE(y_test, y_predict) :
#    return np.sqrt(mean_squared_error(y_test, y_predict))
# rmse = RMSE(y_test, r)
# print('RMSE :', rmse)

r2 = r2_score(y_test, r)
print('r2 스코어 :', r2)

# loss : 3.3282344341278076
# [x]의 결과값 : [[ 9.253285  ]
                # [ 0.75868285]
                # [11.140974  ]
                # [ 1.7025274 ]
                # [ 5.4779067 ]
                # [ 8.309441  ]]
# r2 스코어 : 0.7659835815429688

# 기존 데이터
# r2 스코어 : 0.2135605910836441

# 완벽한 데이터
