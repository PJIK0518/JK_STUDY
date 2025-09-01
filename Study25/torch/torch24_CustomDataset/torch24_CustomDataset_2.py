import torch

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset

# #1. 커스텀 데이터셋 제작

# class MyDataset(Dataset):
#     def __init__(self):
#         super().__init__()
#         self.x =[[1.0], [2.0], [3.0], [4.0], [5.0]]
#         self.y =[0, 1, 0, 1, 0]
    
#     def __len__(self):
#         return len(self.x)
    
#     def __getitem__(self, idx):
#         return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])
    
#     # TensorDataset 보다 자유도가 높음


# #2. 인스턴스 생성


# dataset = MyDataset()
x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = torch.tensor([0, 1, 0, 1, 0])

#. Tensordataset
dataset = TensorDataset(x, y)

#3. DataLoader로 삽입
loader = DataLoader(dataset, batch_size=2, shuffle=True)

#4. 출력
for batch_idx, (x_batch, y_batch) in enumerate(loader):
    print("🔹🔹🔹 batch :", batch_idx, '🔹🔹🔹')
    print("x_bat :", x_batch)
    print("y_bat :", y_batch)
    
# 🔹🔹🔹batch : 0 🔹🔹🔹
# x_bat : tensor([[5.],
#         [3.]])
# y_bat : tensor([0, 0])

# 🔹🔹🔹batch : 1 🔹🔹🔹
# x_bat : tensor([[1.],
#         [4.]])
# y_bat : tensor([0, 1])

# 🔹🔹🔹batch : 2 🔹🔹🔹
# x_bat : tensor([[2.]])
# y_bat : tensor([1])