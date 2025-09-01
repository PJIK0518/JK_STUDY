##################################################
#0. 준비
##################################################
import pandas as pd
import numpy as np

import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

import random

### 랜던고정
RS = 518
random.seed(RS)
np.random.seed(RS)
torch.manual_seed(RS)
torch.cuda.manual_seed(RS)

### GPU
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'CPU'
""" print(DEVICE) cuda:0 """

##################################################
#1. 데이터
##################################################
path = './Study25/_data/kaggle/netflix/'
trn_csv = pd.read_csv(path + 'train.csv')

""" print(trn_csv) [967 rows x 6 columns]
           Date  Open  High  Low    Volume  Close
0    2015-12-16   120   123  118  13181000    123
1    2015-12-17   124   126  122  17284900    123
2    2015-12-18   121   122  118  17948100    118
3    2015-12-21   120   120  116  11670000    117
4    2015-12-22   117   117  115   9689000    116
..          ...   ...   ...  ...       ...    ...
962  2019-10-14   284   287  282   5513200    286
963  2019-10-15   284   286  279   7685600    284
964  2019-10-16   283   288  281  16175900    286
965  2019-10-17   304   309  288  38258900    293
966  2019-10-18   289   291  273  23429900    275 """

""" print(trn_csv.info())
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   Date    967 non-null    object
 1   Open    967 non-null    int64 
 2   High    967 non-null    int64 
 3   Low     967 non-null    int64 
 4   Volume  967 non-null    int64 
 5   Close   967 non-null    int64 
dtypes: int64(5), object(1)
memory usage: 45.5+ KB
None """

""" print(trn_csv.describe())
             Open        High         Low        Volume       Close
count  967.000000  967.000000  967.000000  9.670000e+02  967.000000
mean   223.923475  227.154085  220.323681  9.886233e+06  223.827301
std    104.455030  106.028484  102.549658  6.467710e+06  104.319356
min     81.000000   85.000000   80.000000  1.616300e+06   83.000000
25%    124.000000  126.000000  123.000000  5.638150e+06  124.000000
50%    194.000000  196.000000  192.000000  8.063300e+06  194.000000
75%    329.000000  332.000000  323.000000  1.198440e+07  327.500000
max    421.000000  423.000000  413.000000  5.841040e+07  419.000000 """

import matplotlib.pyplot as plt
data = trn_csv.iloc[:, 1:4]
data['종가'] = trn_csv['Close']

""" print(data) [967 rows x 4 columns] 
     Open  High  Low   종가
0     120   123  118  123
1     124   126  122  123
2     121   122  118  118
3     120   120  116  117
4     117   117  115  116
..    ...   ...  ...  ...
962   284   287  282  286
963   284   286  279  284
964   283   288  281  286
965   304   309  288  293
966   289   291  273  275 """

""" 데이터 분포 확인
hist = data.hist()
plt.show() """

from torch.utils.data.dataset import Dataset, TensorDataset
from torch.utils.data import DataLoader

### 커스텀 데이커 클래스 정의
class Custom_Dataset(Dataset):     
    ## torch 안의 함수이기 때문에 기본적으로 데이터 형태를 torch.float32로 전환함
    def __init__(self, df, timesteps = 30):
        super().__init__()
        self.trn_csv = df
        self.timesteps = timesteps
        
        self.x = self.trn_csv.iloc[:, 1:4].values
        
        # MinMaxScaler 적용
        self.x = (self.x - np.min(self.x, axis=0)) / \
            (np.max(self.x, axis=0) - np.min(self.x, axis=0))
        self.y = self.trn_csv.iloc[:, -1].values
        
        self.x = torch.tensor(self.x, dtype=torch.float32).to(DEVICE)
        self.y = torch.tensor(self.y, dtype=torch.float32).to(DEVICE)
    
    # timestep 부여 (967)
    def __len__(self):
        return len(self.x) - 30 # 전체 데이터 길이 : 마지막 예측값이 없음 
                                # N - timestep // + 1을 하면 x 값은 만들어지는데 y값이 없음

    def __getitem__(self, index):
        x = self.x[index : index+self.timesteps]
        y = self.y[index+self.timesteps]
        return x, y

custon_dataset = Custom_Dataset(trn_csv, 30)

trn_ldr = DataLoader(custon_dataset, batch_size=32)

for batch_idx, (x_bat, y_bat) in enumerate(trn_ldr):
    print(f'🔹🔹🔹🔹🔹 batch : {batch_idx} 🔹🔹🔹🔹🔹')
    print(f'x.shpae : {x_bat.shape}')
    print(f'y.shpae : {y_bat.shape}')
    break

##################################################
#2. 모델 정의
##################################################
### 모델 클래스 설정
class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.rnn = nn.RNN(input_size=3,
                          hidden_size=64,
                          num_layers=3,
                          batch_first=True)
        
        # self.fc1 = nn.Linear(in_features=30.64, out_features=32)
        self.fc1 = nn.Linear(in_features=64, out_features=32)
        self.fc2 = nn.Linear(32, 1)
        self.rlu = nn.ReLU()
        
    def forward(self, x):
        x, _ = self.rnn(x)
        
        # x = x.reshape(-1, 30*64)
        x = x[:,-1,:]
        x = self.fc1(x)
        x = self.rlu(x)
        x = self.fc2(x)
        return x

model = RNN().to(DEVICE)

EPOCH = 1000
##################################################
#3. 컴파일 훈련
##################################################
loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)
"""
def TRAIN(MODL, LOSS, OPTM, LODR):
    MODL.train()
    trn_lss = 0
    
    for XTRN, YTRN in LODR:
        OPTM.zero_grad()
        
        x_trn_prd = MODL(XTRN)
        lss = LOSS(x_trn_prd, YTRN)
        lss.backward()
        OPTM.step()
        trn_lss += lss.item()
    
    return trn_lss / len(LODR)

for e in range(1, EPOCH+1):
    trn_loss = TRAIN(model, loss, optimizer, trn_ldr)
    print('epo :', e)
    print('mse :', trn_loss) """

for epoch in range(1 ,201):
    iterator = tqdm.tqdm(trn_ldr)
    for x, y in iterator:
        optimizer.zero_grad()
        
        hypothesis = model(x)
        
        loss = nn.MSELoss()(hypothesis, y)
        
        loss.backward()
        
        optimizer.step()
        
        iterator.set_description(f'epo : {epoch:>3d}, lss : {loss.item():.5f}')

### save ###
path_save = './Study25/_data/kaggle/netflix/save/'
torch.save(model.state_dict(),path_save + 't25.netflix.pth')

##################################################
#4. 평가예측
##################################################
y_prd = []
y_tru = []
total_lss = 0
with torch.no_grad():
    ### load ###
    model.load_state_dict(torch.load(path_save + 't25.netflix.pth', map_location=DEVICE)) # 기기설정
    
    for x_tst, y_tst in trn_ldr:
        y_pred = model(x_tst.type(torch.FloatTensor).to(DEVICE))
        y_prd.append(y_pred.cpu().numpy())
        y_tru.append(y_tst.cpu().numpy())
        
        lss = nn.MSELoss()(y_pred, y_tst.to(DEVICE))
        total_lss += lss/len(trn_ldr)
        
print(total_lss)

from sklearn.metrics import r2_score
y_prd = np.concatenate(y_prd)
y_tru = np.concatenate(y_tru)

print('R2 :', r2_score(y_tru, y_prd))