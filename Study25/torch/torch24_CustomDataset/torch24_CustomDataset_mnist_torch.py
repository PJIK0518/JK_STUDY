import torch

from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision.datasets import MNIST
from torchvision import transforms

path = './Study25/_data/torch/'

transform = transforms.ToTensor()

trn_dataset = MNIST(path, train = True, download= False, transform= transform)
tst_dataset = MNIST(path, train = False, download= False, transform= transform)

dataset = ConcatDataset([trn_dataset, tst_dataset])

print(dataset)

class MyDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        x, y = self.dataset[index]
        return x, y

DS = MyDataset(dataset)

LD = DataLoader(DS, batch_size=10000, shuffle=True)

for batch, (x_bat, y_bat) in enumerate(LD):
    print("batch :", batch)
    print("x_bat :", x_bat.shape)
    print("y_bat :", y_bat.shape)
    print('🔹🔹🔹🔹🔹🔹🔹🔹🔹')

# batch : 0
# x_bat : torch.Size([10000, 1, 28, 28])
# y_bat : torch.Size([10000])
# 🔹🔹🔹🔹🔹🔹🔹🔹🔹
# batch : 1
# x_bat : torch.Size([10000, 1, 28, 28])
# y_bat : torch.Size([10000])
# 🔹🔹🔹🔹🔹🔹🔹🔹🔹
# batch : 2
# x_bat : torch.Size([10000, 1, 28, 28])
# y_bat : torch.Size([10000])
# 🔹🔹🔹🔹🔹🔹🔹🔹🔹
# batch : 3
# x_bat : torch.Size([10000, 1, 28, 28])
# y_bat : torch.Size([10000])
# 🔹🔹🔹🔹🔹🔹🔹🔹🔹
# batch : 4
# x_bat : torch.Size([10000, 1, 28, 28])
# y_bat : torch.Size([10000])
# 🔹🔹🔹🔹🔹🔹🔹🔹🔹
# batch : 5
# x_bat : torch.Size([10000, 1, 28, 28])
# y_bat : torch.Size([10000])
# 🔹🔹🔹🔹🔹🔹🔹🔹🔹
# batch : 6
# x_bat : torch.Size([10000, 1, 28, 28])
# y_bat : torch.Size([10000])
# 🔹🔹🔹🔹🔹🔹🔹🔹🔹