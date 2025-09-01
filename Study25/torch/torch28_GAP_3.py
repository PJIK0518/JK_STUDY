import torch
import torch.nn as nn

x = torch.randn(1, 64, 10, 10)

gap = nn.AdaptiveMaxPool2d((1,1)) # 가로세로를 몇으로 줄여서 평균 낼 것이냐
#################### 1 ####################
# x = gap(x)

#################### 2 ####################
# x = nn.AdaptiveAvgPool2d((1,1))(x)

#################### 3 ####################
import torch.nn.functional as F
x = F.adaptive_avg_pool2d(x, (1,1))


# print(x.shape) torch.Size([1, 64, 1, 1])

# x = x.view(x.size(0), -1)
x = torch.flatten(x, 1)

# print(x.shape) torch.Size([1, 64])
# print(x.shape) torch.Size([1, 64])