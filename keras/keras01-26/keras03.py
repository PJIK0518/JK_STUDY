# keras02 복붙 후 수정 + a

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,4,5,6])

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))

epochs = 1000
    # 모델 구성에 epochs 값을 빼서 따로 볼수도 있음
    # 이때는 뒤에서 부터는 그냥 epochs를 써주면 1000으로 인식 or 표기 가능
    # 이때는 표기를 자동화해서 빼주기도 가능
    
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=epochs)

print('##########################################################################################')

#4. 평가, 예측
loss = model.evaluate(x, y)
results = model.predict([7])

print('epochs : ', epochs)
print('로스 : ', loss)
print('예측값 : ', results)