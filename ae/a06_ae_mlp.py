# aa04_ae2_그림.py - capy
# 그림 그리는 plt 쪽만 바꿔보자 학습하며 이미지 변환한 순서대로 한꺼본에 뽑아 낼 거다 

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input


# 1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28*28).astype('float32') / 255

x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)

print(x_train_noised.shape, x_test_noised.shape) # (60000, 784) (10000, 784)
# exit()
print(np.max(x_train), np.min(x_test))
print(np.max(x_train_noised), np.min(x_test_noised))
# exit()
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, 0, 1)

print(np.max(x_train_noised), np.min(x_test_noised)) # 1.0 0.0
# exit()

# 2. 모델
input_img = Input(shape=(28*28))

####[실습] : 딥하게 구성####

def autoencoder(a,b,c,d,e):
    model = Sequential()
    model.add(Dense(units=a, input_shape=(28*28,))) 
    model.add(Dense(b)) 
    model.add(Dense(c)) 
    model.add(Dense(d))
    model.add(Dense(e))  
    model.add(Dense(784, activation='sigmoid'))
    return model

hidden_size = [
     ['모래시계형', 128, 64, 31, 64, 128],
     ['다이아몬드형', 64, 128, 256, 128, 64],
     ['통나무형', 128, 128, 128, 128, 128],
]

names = []
rlt_list = []

names.append('원본')
rlt_list.append(x_test)

for name, a, b, c, d, e in hidden_size:
     print(f'🔹🔹🔹🔹🔹🔹 {name} 🔹🔹🔹🔹🔹🔹')
     model = autoencoder(a,b,c,d,e)

     model.compile(optimizer='adam', loss='binary_crossentropy')
     model.fit(x_train_noised, x_train, epochs=20, batch_size=128, validation_split=0.2, verbose=0)

     decoded_imgs = model.predict(x_test_noised)
     
     rlt_list.append(decoded_imgs)
     names.append(f'복원({name})')
     
rlt_list = np.array(rlt_list)

print(rlt_list.shape)
exit()

import matplotlib.pyplot as plt
import random

num_rows = len(rlt_list)         # 원본/노이즈/복원들
num_cols = 5                    # 샘플 개수
random_images = random.sample(range(x_test.shape[0]), num_cols)

fig, axes = plt.subplots(num_rows, num_cols, figsize=(3*num_cols, 3*num_rows))

# axes가 1차원/2차원 모두 처리
if num_rows == 1:
    axes = [axes]              # 1행일 때도 반복문 통일을 위해 리스트로
    
for r, row_axes in enumerate(axes):
    for c, ax in enumerate(np.atleast_1d(row_axes)):
        ax.imshow(rlt_list[r][random_images[c]].reshape(28, 28), cmap='gray')
        if c == 0:
            ax.set_title(names[r])
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
plt.show()