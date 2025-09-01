# 49-2 복사

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt

Augment_size = 100

(x_trn, y_trn), (x_tst, y_tst) = fashion_mnist.load_data()

# 2025-06-17 오전 새로운 내용!!!!!
# 2차원 데이터를 우선 1차원으로 펼친 후 반복하면, 1차원 데이터인 겻은 그대로인데 원소 수가 100배가 된다.
# 그 후에 reshape(-1, 28, 28, 1)로 변경하면 데이터 순서가 꼬이지 않고 형태 변환을 할 수 있다.
aaa = np.tile(x_trn[0].reshape(28*28), Augment_size).reshape(-1,28,28,1)
# aaa = np.tile(x_trn[0], (Augment_size, 1)).reshape(-1,28,28,1)

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
# y 데이터 생략 시 x 데이터만 numpy array 형태로 반환
x_data = IDG.flow(
    aaa,                         # x 데이터
    # np.zeros(Augment_size),      # y 데이터 생성, Augment_size 만큼 0 생성
    batch_size=Augment_size,     # 통배치      
    shuffle=False,
).next()   

print(len(x_data))      # x 데이터의 행 개수 100
print(x_data[0].shape)  # 첫번째 이미지 (28, 28, 1)
print(x_data[1].shape)  # 두번째 이미지 (28, 28, 1)

plt.figure(figsize=(7,7))

for i in range(49):
    plt.subplot(7, 7, i+1)
    plt.imshow(x_data[i], cmap='gray') # y데이터가 없기에 x_data[0][i] 대신 x_data[i]
    plt.axis('off')

plt.show()

