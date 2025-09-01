from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img           # 이미지 불러오기
from tensorflow.keras.preprocessing.image import img_to_array       # 불러온 이미지의 수치화
import matplotlib.pyplot as plt
import numpy as np

path = 'c:/Study25/_data/image/me/'

img = load_img(path + 'me.PNG', target_size=(150,150))

# print(img) <PIL.Image.Image image mode=RGB size=100x100 at 0x1567563C940>
# PIL = Python Image Library

#####################
# 사진 출력
# plt.imshow(img)
# plt.show()

#####################
# 사진의 수치화
arr = img_to_array(img)
# print(arr)
# print(arr.shape) (100, 100, 3)
# print(type(arr)) <class 'numpy.ndarray'>

#####################
# 모델과 차원 맞추기
#1) reshape
# arr = arr.reshape(1,100,100,3)
print(arr)
print(arr.shape)

#2) np.expand_dims
ed = np.expand_dims(arr, axis=0) # aixs의 위치에 차원 1을 추가 시킴 = 대괄호 하나를 추가시킨다
print(ed)
print(ed.shape)

#####################
# numpy 파일 생성

np.save(path + 'keras47_me.npy', arr=ed)