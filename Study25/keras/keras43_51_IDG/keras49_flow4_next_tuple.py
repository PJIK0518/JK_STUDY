# 49-3 복사

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt

Augment_size = 100

(x_trn, y_trn), (x_tst, y_tst) = fashion_mnist.load_data()

aaa = np.tile(x_trn[0].reshape(28*28), Augment_size).reshape(-1,28,28,1)

IDG = ImageDataGenerator(
    rescale=1./255.,
    horizontal_flip=True,   # 좌우 반전
    # vertical_flip=True,
    width_shift_range=0.1,
    # height_shift_range=0.1,
    rotation_range=15,
    # zoom_range=0.2,
    # shear_range=0.7,
    fill_mode='nearest' 
)

# 2025-06-17 오전 새로운 내용!!!!!
# y값도 입력하면 x, y데이터가 튜플로 묶여서 반환된다.
xy_data = IDG.flow(
    aaa,                         # x 데이터
    np.zeros(Augment_size),      # y 데이터 생성, Augment_size 만큼 0 생성
    batch_size=Augment_size,     # 통배치      
    shuffle=False,
).next()   

x_data, y_data = xy_data # 튜플에 담겨있는 x, y데이터를 각각 분리

print(x_data.shape) # (100, 28, 28, 1)
print(y_data.shape) # (100, )

# print(len(xy_data))      # 2, 튜플의 크기 (x, y)
# print(xy_data[0].shape)  # (100, 28, 28, 1), xdata의 크기
# print(xy_data[1].shape)  # (100, ), y_data의 크기

plt.figure(figsize=(7,7))

for i in range(49):
    plt.subplot(7, 7, i+1)
    # plt.imshow(x_data[i], cmap='gray')
    plt.axis('off')

plt.show()

