# 100 100 3 의 이미지를 
# 10 10 11 으로 줄여봐

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D

# model = Sequential()
# model.add(Conv2D(11, 3, input_shape=(100,100,3)))
# model.add(Conv2D(11, 3, input_shape=(100,100,3)))
# model.add(Conv2D(11, 3, input_shape=(100,100,3)))
# model.add(Conv2D(11, 3, input_shape=(100,100,3)))
# model.add(Conv2D(11, 3, input_shape=(100,100,3)))
# model.add(Conv2D(11, 3, input_shape=(100,100,3)))
# model.add(Conv2D(11, 3, input_shape=(100,100,3)))
# model.add(Conv2D(11, 3, input_shape=(100,100,3)))
# model.add(Conv2D(11, 3, input_shape=(100,100,3)))
# model.add(Conv2D(11, 3, input_shape=(100,100,3)))
# model.add(Conv2D(11, 3, input_shape=(100,100,3)))
# model.add(Conv2D(11, 3, input_shape=(100,100,3)))
# model.add(Conv2D(11, 3, input_shape=(100,100,3)))
# model.add(Conv2D(11, 3, input_shape=(100,100,3)))
# model.add(Conv2D(11, 3, input_shape=(100,100,3)))
# model.add(Conv2D(11, 3, input_shape=(100,100,3)))
# model.add(Conv2D(10, 3, input_shape=(100,100,3)))
# model.add(Conv2D(10, 3, input_shape=(100,100,3)))
# model.add(Conv2D(10, 3, input_shape=(100,100,3)))
# model.add(Conv2D(10, 3, input_shape=(100,100,3)))
# model.add(Conv2D(10, 3, input_shape=(100,100,3)))
# model.add(Conv2D(10, 3, input_shape=(100,100,3)))
# model.add(Conv2D(10, 3, input_shape=(100,100,3)))
# model.add(Conv2D(10, 3, input_shape=(100,100,3)))
# model.add(Conv2D(10, 3, input_shape=(100,100,3)))
# model.add(Conv2D(10, 3, input_shape=(100,100,3)))
# model.add(Conv2D(10, 3, input_shape=(100,100,3)))
# model.add(Conv2D(10, 3, input_shape=(100,100,3)))
# model.add(Conv2D(10, 3, input_shape=(100,100,3)))
# model.add(Conv2D(10, 3, input_shape=(100,100,3)))
# model.add(Conv2D(10, 3, input_shape=(100,100,3)))
# model.add(Conv2D(10, 3, input_shape=(100,100,3)))
# model.add(Conv2D(10, 3, input_shape=(100,100,3)))
# model.add(Conv2D(10, 3, input_shape=(100,100,3)))
# model.add(Conv2D(10, 3, input_shape=(100,100,3)))
# model.add(Conv2D(10, 3, input_shape=(100,100,3)))
# model.add(Conv2D(10, 3, input_shape=(100,100,3)))
# model.add(Conv2D(10, 3, input_shape=(100,100,3)))
# model.add(Conv2D(10, 3, input_shape=(100,100,3)))
# model.add(Conv2D(10, 3, input_shape=(100,100,3)))
# model.add(Conv2D(10, 3, input_shape=(100,100,3)))
# model.add(Conv2D(10, 3, input_shape=(100,100,3)))
# model.add(Conv2D(10, 3, input_shape=(100,100,3)))
# model.add(Conv2D(10, 3, input_shape=(100,100,3)))
# model.add(Conv2D(11, 3, input_shape=(100,100,3)))
# model.summary()

""" model.summary()
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 10, 10, 11)        273284
=================================================================
Total params: 273,284
Trainable params: 273,284
Non-trainable params: 0 """

model = Sequential()
model.add(Conv2D(11, 2, input_shape=(100,100,3)))
model.add(MaxPool2D())
model.add(Conv2D(11, 2))
model.add(MaxPool2D())
model.add(Conv2D(11, 2))
model.add(MaxPool2D())
model.add(Conv2D(11, 2))
model.summary()