import numpy as np

a = np.array([[ 1, 2, 3, 4, 5, 6, 7, 8, 9,10],
              [ 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
              ]).T

print(a)

def split(DS, TS):
    xy_a = []
    for i in range(len(DS) - TS):
        xy = DS[i : (i + TS + 1),:]
        xy_a.append(xy)
        A = np.array(xy_a)
        y = A[:,-1].T
    return A[:, :-1, :-1], y[1].reshape(-1,1)
x, y = split(a, 4)

# print(x)
# print(y)
# print(x.shape) (6, 4, 2)
# print(y.shape) (6, 1)