import numpy as np

a = np.array(range(1, 11))
TimeStep = 4

# print(a)        [ 1  2  3  4  5  6  7  8  9 10]
# print(a.shape)  (10,)

#######################################
### 시계열 데이터 제작 함수 _ 버전 SS ###

def split_X(datasets, TimeStep):
    xy_a = []
    for i in range(len(datasets) - TimeStep + 1):        # x, y 가 합쳐진 형태를 만들기 때문에 끝까지 사용
        subset = datasets[i : (i + TimeStep)]
        xy_a.append(subset)
    return np.array(xy_a)

A = split_X(a, TimeStep)
x = A[:,:-1]
y = A[:,-1]

print(A)

#######################################
### 시계열 데이터 제작 함수 _ 버전 JK ###

# def split(DS, TS):
#     xy_a = []
#     for i in range(len(DS) - TS):
#         xy = DS[i : (i + TS + 1)]
#         xy_a.append(xy)
#         A = np.array(xy_a)
#     return A[:,:-1], A[:,-1]
# x, y = split(a, TimeStep)

print(x)
print(y)
print(x.shape)
print(y.shape)

""" print(A)
[[ 1  2  3  4  5]
 [ 2  3  4  5  6]
 [ 3  4  5  6  7]
 [ 4  5  6  7  8]
 [ 5  6  7  8  9]
 [ 6  7  8  9 10]]
"""
# print(x)      # [[1 2 3 4]
                #  [2 3 4 5]
                #  [3 4 5 6]
                #  [4 5 6 7]
                #  [5 6 7 8]
                #  [6 7 8 9]] 
# print(y)         [ 5  6  7  8  9 10] 
# print(x.shape)   (6, 4) 
# print(y.shape)   (6,) 