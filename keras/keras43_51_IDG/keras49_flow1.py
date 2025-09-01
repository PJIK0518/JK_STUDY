# 47.copy

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
# 데이터 증폭
IDG = ImageDataGenerator(
    rescale=1./255.,
    horizontal_flip=True,         # 좌우반전
    # vertical_flip=True,         # 상하반전
    width_shift_range=0.1,        # 평행이동 10%
    height_shift_range=0.1,         
    rotation_range=15,            # 회전 15도
    zoom_range=0.2,
    shear_range=0.7,
    fill_mode='nearest' 
)


IT = IDG.flow(ed,                  # flow : 폴더에서 가져오는 것이 아니라 
    batch_size=1000)

""" print(IT) <keras.preprocessing.image.NumpyArrayIterator object at 0x000001D826095730> """
print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print(IT)   # <keras.preprocessing.image.NumpyArrayIterator object at 0x000001D826095730>
print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')

# aaa = IT.next()     # python 2.0 문법
# print(aaa)
# print(aaa.shape)    (1, 150, 150, 3)

# bbb = next(IT)
# print(bbb)
# print(bbb.shape)    (1, 150, 150, 3)

# print(IT.next())    # 기본적으로 Iterator의 데이터보다 많으면 ERROR
# print(IT.next())    # NumpyArrayIterator의 경우 데이터보다 많이 하면 순환되게 설정...!
# print(IT.next())


#####################
# 증폭 그림 불러오기

fig, ax = plt.subplots(nrows=5, ncols=5, figsize=(5,5))

for i in range(5):
    for j in range(5):
        batch = IT.next()
        batch = next(IT)
        # print(batch.shape)
        batch = batch.reshape(150,150,3)    # 이미지 데이터 1개 기준으로 reshape (1,150,150,3) > (150,150,3)
        
        ax[i, j].imshow(batch)
        ax[i, j].axis('off')                # Fig의 x, y 축 on-off

plt.tight_layout()                          # Fig 사이의 여백을 짧게 설정
plt.show()
    








#####################
# numpy 파일 생성
# np.save(path + 'keras47_me.npy', arr=ed)
