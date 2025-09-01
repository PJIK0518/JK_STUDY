import torch

x = torch.tensor([
    [[1,2],[3,4],[5,6]],
    [[7,8],[9,10],[11,12]],
])

# print(x.shape) torch.Size([2, 3, 2])

x = x[:, -1, :]
print(x.shape)

##### [실습] #####
# [[tensroslice]] : 하나만 남기면 자동으로 차원 축소도 됨
# x = x[:, :-1, :]
# tensor([[[ 1,  2],[ 3,  4]],
#         [[ 7,  8],[ 9, 10]]])

# x = x[:, 0, :]
# tensor([[1, 2],
#         [7, 8]])

# x = x[:, -1, :]
# tensor([[ 5,  6],
#         [11, 12]])