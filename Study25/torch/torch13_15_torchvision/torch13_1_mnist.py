##################################################
#0. 준비
##################################################
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

### 랜덤고정
RS = 55
torch.cuda.manual_seed(RS)
torch.manual_seed(RS)
np.random.seed(RS)
random.seed(RS)

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'CPU')
print('torch :', torch.__version__)
print('dvice :', DEVICE)

##################################################
#1. 데이터
##################################################
from torchvision.datasets import MNIST

path = './Study25/_data/torch/'

trn_dataset = MNIST(path, train = True, download= True)
tst_dataset = MNIST(path, train = False, download= True)

""" print(trn_dataset) """
""" print(type(trn_dataset)) <class 'torchvision.datasets.mnist.MNIST'> """
""" print(trn_dataset[0]) (<PIL.Image.Image image mode=L size=28x28 at 0x74F239421090>, 5) """
                            # PIL : python image library
x_trn, y_trn = trn_dataset.data/255., trn_dataset.targets
x_tst, y_tst = tst_dataset.data/255., tst_dataset.targets

""" print(x_trn.shape, x_trn.size()) torch.Size([60000, 28, 28]) torch.Size([60000, 28, 28]) """
""" print(np.min(x_trn.numpy()), np.max(x_trn.numpy())) 0.0 1.0 """

x_trn, x_tst = x_trn.view(-1, 28*28), x_tst.view(-1, 28*28)
    # view : torch에서 제공되는 reshape
        #   약간의 문제가 있다고는 하지만 빠르다
""" print(x_trn.size()) torch.Size([60000, 784]) """
""" print(x_tst.size()) torch.Size([10000, 784]) """

### Dataloader
from torch.utils.data import TensorDataset # 데이터 합치기
from torch.utils.data import DataLoader    # Batch 나누기
trn_Dataset = TensorDataset(x_trn, y_trn)
tst_Dataset = TensorDataset(x_tst, y_tst)

trn_Loader = DataLoader(trn_Dataset, batch_size=32, shuffle=True)
tst_Loader = DataLoader(tst_Dataset, batch_size=32, shuffle=False)

##################################################
#2. 모델
##################################################
class DNN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
    # = super(self, DNN).__init__()
    
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU())
        
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU())
        
        self.hidden_layer3 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU())
        
        self.hidden_layer4 = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU())
        
        self.output_layer = nn.Linear(16, 10)
    
    def forward(self, x):
        x =self.hidden_layer1(x)
        x =self.hidden_layer2(x)
        x =self.hidden_layer3(x)
        x =self.hidden_layer4(x)
        x =self.output_layer(x)
        
        return x

model = DNN(28*28).to(DEVICE)

EPOCHS = 100

##################################################
#3. 컴파일 훈련
##################################################
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.1e-4) # 0.0001

def TRAIN(MODL, LOSS, OPTM, LODR):
    MODL.train()
    epo_lss = 0
    epo_acc = 0
    
    for XTRN, YTRN in LODR:
        XTRN, YTRN = XTRN.to(DEVICE), \
                     YTRN.to(DEVICE)
        
        OPTM.zero_grad()
        
        x_trn_prd = MODL(XTRN)
        trn_loss = LOSS(x_trn_prd, YTRN)
        
        trn_loss.backward()     # 기울기 계산
        OPTM.step()             # w = w - lr * 기울기
        
        YPRD = torch.argmax(x_trn_prd, 1)
        acc = (YPRD == YTRN).float().mean()
        
        epo_lss += trn_loss.item()
        epo_acc += acc
    
    return epo_lss / len(LODR), epo_acc / len(LODR)

def EVALUATE(MODL, LOSS, LODR):
    MODL.eval()
    total_loss = 0
    total_acc = 0
    
    for XVAL, YVAL in LODR:
        XVAL, YVAL = XVAL.to(DEVICE), YVAL.to(DEVICE)
        with torch.no_grad():
            YPRD = MODL(XVAL)
            loss = LOSS(YPRD, YVAL)
            total_loss += loss.item()
            
            YPRD_label = torch.argmax(YPRD, dim=1)
            acc = (YPRD_label == YVAL).float().mean()
            total_acc += acc.item()
    
    return total_loss / len(LODR), total_acc / len(LODR)

for e in range(1, EPOCHS+1):
    trn_loss, trn_acc = TRAIN(model, loss, optimizer, trn_Loader)
    val_loss, val_acc = EVALUATE(model, loss, tst_Loader)
    print(f'epo : {e}')
    print(f'trn_lss : {trn_loss:.5f} | trn_acc : {trn_acc:.5f}')
    print(f'val_lss : {val_loss:.5f} | val_acc : {val_acc:.5f}')

############################################################
#4. 평가 예측
############################################################
tst_loss = EVALUATE(model, loss, tst_Loader)

x_tst_prd = model(x_tst)

# print(type(x_tst_prd)) <class 'torch.Tensor'> 
# :: x_tst_prd 및 y_tst는 torch.Tensor 형태 
# :: accuracy_score는 numpy 형태로 cpu에서 계산 

y_tst = y_tst.detach().cpu().numpy()
x_tst_prd = np.round(x_tst_prd.detach().cpu().numpy())

acc = accuracy_score(y_tst, x_tst_prd)

print(f'tst_lss : {tst_loss:.5f}')
print(f'tst_acc : {acc:.5f}')

# bce : 0.7100207209587097
# acc : 0.9707602339181286