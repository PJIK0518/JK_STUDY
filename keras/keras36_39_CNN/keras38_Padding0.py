from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense

#2. 모델구성
model = Sequential()
model.add(Conv2D(10, (3,3), input_shape=(10,10,1),
                 strides = 1,
                 padding = 'same',
                #  padding = 'valid',
))
model.add(Conv2D(filters=9, kernel_size=(3,3),
                 strides=1,
                 padding='valid'))       # padding의 default, 소실 데이터 안 만들고 그냥 진행
model.add(Conv2D(18, 4))                 # kernel_size는 통상적으로 정사각형(n,n)으로 사용, 4로 표기하면 (4,4)로 인식
                                         # 이미지 분석을 위해 Conv2D를 몇 번 먹이면 데이터가 과도하게 압축되기 때문에 Paddging로 이러한 소실을 방지해서 연산량을 늘릴 수 있음
model.summary()
""" model.summary()
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 10, 10, 10)        50        
 conv2d_1 (Conv2D)           (None, 8, 8, 9)           819
 conv2d_2 (Conv2D)           (None, 5, 5, 18)           1160
=================================================================
Total params: 2,029
Trainable params: 2,029
Non-trainable params: 0 """