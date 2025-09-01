from torch.utils.data import DataLoader, Dataset
from keras.datasets import mnist

import numpy as np
import torch

(x_trn, y_trn), (x_tst, y_tst) = mnist.load_data()

x = np.concatenate([x_trn, x_tst], axis=0)
y = np.concatenate([y_trn, y_tst], axis=0)

class Mydataset(Dataset):
    def __init__(self, x ,y):
        super().__init__()
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

DS = Mydataset(x, y)
LD = DataLoader(DS, batch_size=10000, shuffle=False)

for batch, (x, y) in enumerate(LD):
    print(f'🔹🔹🔹🔹🔹 {batch}🔹🔹🔹🔹🔹')
    print(x.shape)
    print(y.shape)

# 🔹🔹🔹🔹🔹 0🔹🔹🔹🔹🔹
# torch.Size([10000, 28, 28])
# torch.Size([10000])
# 🔹🔹🔹🔹🔹 1🔹🔹🔹🔹🔹
# torch.Size([10000, 28, 28])
# torch.Size([10000])
# 🔹🔹🔹🔹🔹 2🔹🔹🔹🔹🔹
# torch.Size([10000, 28, 28])
# torch.Size([10000])
# 🔹🔹🔹🔹🔹 3🔹🔹🔹🔹🔹
# torch.Size([10000, 28, 28])
# torch.Size([10000])
# 🔹🔹🔹🔹🔹 4🔹🔹🔹🔹🔹
# torch.Size([10000, 28, 28])
# torch.Size([10000])
# 🔹🔹🔹🔹🔹 5🔹🔹🔹🔹🔹
# torch.Size([10000, 28, 28])
# torch.Size([10000])
# 🔹🔹🔹🔹🔹 6🔹🔹🔹🔹🔹
# torch.Size([10000, 28, 28])
# torch.Size([10000])