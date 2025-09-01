import numpy as np

aaa = np.array([[-10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50],
                [100, 200, -30, 400, 500, 600, -70000, 800, 900, 1000, 210, 420, 350]
                ]).T


def outlier(data):
    out = []
    up = []
    low = []
    for i in range(data.shape[1]):
        col = data[:, i]
        Q1, Q3 = np.percentile(col, [25, 75])
        
        IQR = Q3 - Q1
        print('IQR :', IQR)
        
        lower_bound = Q1 - (1.5 * IQR)
        upper_bound = Q3 + (1.5 * IQR)
        
        out_i = np.where((col > upper_bound) | (col < lower_bound))[0]
        out.append(out_i)
        up.append(upper_bound)
        low.append(lower_bound)
    return out, up, low

OUT, UP, LOW = outlier(aaa)

print(OUT) # [array([ 0, 12]), array([6])]
print(UP)  # [19.0, 1200.0]
print(LOW) # [-5.0, -400.0]

import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].boxplot(aaa[:,0])
axs[0].axhline(UP[0], color = 'red', label = 'upper_bound')
axs[0].axhline(LOW[0], color = 'red', label = 'lower_bound')
axs[0].legend()

axs[1].boxplot(aaa[:,1])
axs[1].axhline(UP[1], color = 'blue', label = 'upper_bound')
axs[1].axhline(LOW[1], color = 'blue', label = 'lower_bound')
axs[1].legend()

plt.tight_layout()
plt.show()

### 컬럼 별로 이상치를 확인하는게 기본이긴 하다...!
### But : 컬럼 간의 관계에서도 이상치를 확인 가능
### ex  : 몸무게  30    130
       #  키     190    110