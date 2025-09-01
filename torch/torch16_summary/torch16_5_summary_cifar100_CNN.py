##################################################
#0. 준비
##################################################
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score

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
from torchvision.datasets import CIFAR100

import torchvision.transforms as tr

trsf = tr.Compose([tr.Resize(56), tr.ToTensor(), tr.Normalize((0.5), (0.5))])


path = './Study25/_data/torch/'

trn_DS = CIFAR100(path, train=True, download=True, transform=trsf)
tst_DS = CIFAR100(path, train=False, download=True, transform=trsf)

trn_dl = DataLoader(trn_DS, batch_size=1000, shuffle=True)
tst_dl = DataLoader(tst_DS, batch_size=1000, shuffle=False)

##################################################
#2. 모델
##################################################
class DNN(nn.Module):
    def __init__(self, features):
        super().__init__()
        
        self.seq1 = nn.Sequential(
            nn.Conv2d(features, 64, kernel_size=(3,3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.2)
        )
        
        self.seq2 = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.seq3 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.outp = nn.Linear(128, 100)
        
    def forward(self, x):
        x = self.seq1(x)
        x = self.seq2(x)
        x = self.seq3(x)
        x = self.outp(x)
        
        return x

model = DNN(3).to(DEVICE)

EPOCHS = 100

from torchsummary import summary
summary(model, (1,56,56))
from torchinfo import summary
summary(model, (32,1,56,56))
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
    trn_lss, trn_acc = TRAIN(model, loss, optimizer, trn_dl)
    val_lss, val_acc = EVALUATE(model, loss, tst_dl)
    print(f'epo : {e}')
    print(f'trn_lss : {trn_lss:.5f} | trn_acc : {trn_acc:.5f}')
    print(f'val_lss : {val_lss:.5f} | val_acc : {val_acc:.5f}')
    

############################################################
#4. 평가 예측
############################################################
tst_lss, tst_acc = EVALUATE(model, loss, tst_dl)

print(f'tst_lss : {tst_lss:.5f}')
print(f'tst_acc : {tst_acc:.5f}')

# tst_lss : 3.21621
# tst_acc : 0.22690

# z_scorenormalization