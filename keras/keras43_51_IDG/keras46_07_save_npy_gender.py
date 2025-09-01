# 46_01.copy

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

#####################################
### IDG 인스턴스

trn_IDG = ImageDataGenerator(rescale = 1./255.)
tst_IDG = ImageDataGenerator(rescale = 1./255.)

#####################################
### Image to DI

path_trn = 'C:/Study25/_data/kaggle/men_women/train/'

xy_trn = trn_IDG.flow_from_directory(path_trn,
                                     target_size=(200,200),
                                     batch_size=100,
                                     color_mode='rgb',
                                     class_mode='binary',
                                     shuffle=True,
                                     seed=50)

""" path_tst = 'C:/Study25/_data/kaggle/men_women/'

xy_tst = trn_IDG.flow_from_directory(path_tst,
                                     target_size=(150,150),
                                     batch_size=100,
                                     color_mode='rgb',
                                     class_mode='binary') """


print(xy_trn.class_indices)

#####################################
### DI to list

all_x_trn = []
all_y_trn = []
all_x_tst = []
all_y_tst = []

for i in range(len(xy_trn)):
    x_trn_batchs, y_trn_batchs = xy_trn[i]
    all_x_trn.append(x_trn_batchs)
    all_y_trn.append(y_trn_batchs)

""" for i in range(len(xy_tst)):
    x_tst_batchs, y_tst_batchs = xy_tst[i]
    all_x_tst.append(x_tst_batchs)
    all_y_tst.append(y_tst_batchs) """
    
#####################################
### List to Numpy

x_trn = np.concatenate(all_x_trn, axis=0)
y_trn = np.concatenate(all_y_trn, axis=0)
""" x_tst = np.concatenate(all_x_tst, axis=0) """


# print(type(x_trn), type(y_trn)) <class 'numpy.ndarray'>
# print(x_trn.shape) (3309, 150, 150, 3)
# print(y_trn.shape) (3309,)
""" print(x_tst.shape) """


path_NP = 'C:/Study25/_data/kaggle/men_women/'

np.save(path_NP + 'x_trn.npy', arr = x_trn)
np.save(path_NP + 'y_trn.npy', arr = y_trn)
""" np.save(path_NP + 'x_tst.npy', arr = x_tst) """