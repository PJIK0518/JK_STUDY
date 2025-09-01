#0. 준비

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score

import torch.optim as optim
import torch.nn as nn
import torch

import random
import numpy as np

RS = 555
torch.cuda.manual_seed(RS)
torch.manual_seed(RS)
np.random.seed(RS)
random.seed(RS)

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'CPU')

##################################################
#1. 데이터
##################################################
from torchvision.datasets import FashionMNIST
import torchvision.transforms as tr

trsf = tr.Compose([tr.Resize(56), tr.ToTensor(), tr.Normalize((0.5), (0.5))])

path = './Study25/_data/torch/'

trn_DS = FashionMNIST(path, train=True, download=True, transform=trsf)
tst_DS = FashionMNIST(path, train=False, download=True, transform=trsf)

trn_ld = DataLoader(trn_DS, batch_size=1000, shuffle=True)
tst_ld = DataLoader(trn_DS, batch_size=1000, shuffle=False)

##################################################
#2. 모델
##################################################
class CNN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.seq1 = nn.Sequential(
            nn.Conv2d(num_features, 64, kernel_size=(3,3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.2)
        )
        
        self.seq2 = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.seq3 = nn.Sequential(
            nn.Linear(32, 16),
        )
        
        self.seq4 = nn.Sequential(
            nn.Linear(16, 16))

        self.outp = nn.Linear(16,10)
    
    def forward(self, x):
        x = self.seq1(x)
        x = self.seq2(x)
        x = self.seq3(x)
        x = self.seq4(x)
        x = self.outp(x)
        
        return x
    
model = CNN(1).to(DEVICE)

EPOCHS = 100

""" from torchsummary import summary
summary(model, (1,56,56))
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 54, 54]             640
              ReLU-2           [-1, 64, 54, 54]               0
         MaxPool2d-3           [-1, 64, 27, 27]               0
           Dropout-4           [-1, 64, 27, 27]               0
           Flatten-5                [-1, 46656]               0
            Linear-6                   [-1, 32]       1,493,024
              ReLU-7                   [-1, 32]               0
           Dropout-8                   [-1, 32]               0
            Linear-9                   [-1, 16]             528
           Linear-10                   [-1, 16]             272
           Linear-11                   [-1, 10]             170
================================================================
Total params: 1,494,634
Trainable params: 1,494,634
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 3.92
Params size (MB): 5.70
Estimated Total Size (MB): 9.63
---------------------------------------------------------------- """
""" from torchinfo import summary
summary(model, (32,1,56,56))
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
CNN                                      [32, 10]                  --
├─Sequential: 1-1                        [32, 64, 27, 27]          --
│    └─Conv2d: 2-1                       [32, 64, 54, 54]          640
│    └─ReLU: 2-2                         [32, 64, 54, 54]          --
│    └─MaxPool2d: 2-3                    [32, 64, 27, 27]          --
│    └─Dropout: 2-4                      [32, 64, 27, 27]          --
├─Sequential: 1-2                        [32, 32]                  --
│    └─Flatten: 2-5                      [32, 46656]               --
│    └─Linear: 2-6                       [32, 32]                  1,493,024
│    └─ReLU: 2-7                         [32, 32]                  --
│    └─Dropout: 2-8                      [32, 32]                  --
├─Sequential: 1-3                        [32, 16]                  --
│    └─Linear: 2-9                       [32, 16]                  528
├─Sequential: 1-4                        [32, 16]                  --
│    └─Linear: 2-10                      [32, 16]                  272
├─Linear: 1-5                            [32, 10]                  170
==========================================================================================
Total params: 1,494,634
Trainable params: 1,494,634
Non-trainable params: 0
Total mult-adds (M): 107.53
==========================================================================================
Input size (MB): 0.40
Forward/backward pass size (MB): 47.79
Params size (MB): 5.98
Estimated Total Size (MB): 54.17
========================================================================================== """
exit()

##################################################
#3. 컴파일 훈련
##################################################
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.0005)

### [함수] 훈련
def TRAIN(MODL, LOSS, OPTM, LODR):
    MODL.train()
    epo_lss = 0
    epo_acc = 0
    
    for XTRN, YTRN in LODR:
        XTRN, YTRN = XTRN.to(DEVICE), YTRN.to(DEVICE)
        
        OPTM.zero_grad()
        x_trn_prd = MODL(XTRN)
        trn_loss = LOSS(x_trn_prd, YTRN)
        
        trn_loss.backward()
        OPTM.step()
        
        YPRD = torch.argmax(x_trn_prd, 1)
        acc = (YTRN == YPRD).float().mean()
        
        epo_lss += trn_loss.item()
        epo_acc += acc.item()
    
    return epo_lss / len(LODR), epo_acc / len(LODR)

### [함수] 평가
def EVALUATE(MODL, LOSS, LODR):
    MODL.eval()
    total_lss = 0
    total_acc = 0
    
    for XTST, YTST in LODR:
        XTST, YTST = XTST.to(DEVICE), YTST.to(DEVICE)
        with torch.no_grad():
            x_tst_prd = MODL(XTST)
            tst_loss = LOSS(x_tst_prd, YTST)
            total_lss += tst_loss.item()
            
            YPRD = torch.argmax(x_tst_prd, 1)
            acc = (YTST == YPRD).float().mean()
            total_acc += acc.item()
            
    return total_lss / len(LODR), total_acc/ len(LODR)

for e in range(1, EPOCHS+1):
    trn_lss, trn_acc = TRAIN(model, loss, optimizer, trn_ld)
    val_lss, val_acc = EVALUATE(model, loss, tst_ld)
    print(f'epo : {e}')
    print(f'trn_lss : {trn_lss:.5f} | trn_acc : {trn_acc:.5f}')
    print(f'val_lss : {val_lss:.5f} | val_acc : {val_acc:.5f}')
    

############################################################
#4. 평가 예측
############################################################
tst_lss, tst_acc = EVALUATE(model, loss, tst_ld)

print(f'tst_lss : {tst_lss:.5f}')
print(f'tst_acc : {tst_acc:.5f}')

# tst_lss : 0.33618
# tst_acc : 0.89230

# tst_lss : 0.09666
# tst_acc : 0.96638

# z_scorenormalization
# tst_lss : 0.06207
# tst_acc : 0.97832