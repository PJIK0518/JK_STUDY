##################################################
#0. 준비
##################################################
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

import random

### 랜덤고정
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
path = './Study25/_data/kaggle/jena/'
trn_csv = pd.read_csv(path + 'x_trn_0620_8.csv')

""" print(trn_csv) [420336 rows x 5 columns]
        p (mbar)  T (degC)    rh (%)  rho (g/m**3)  wv (m/s)
0       0.814939 -0.215129  1.050218      2.294889      1.03
1       0.815430 -0.225590  1.056300      2.346221      0.72
2       0.815037 -0.228273  1.086709      2.357239      0.19
3       0.814840 -0.222908  1.104954      2.330947      0.34
4       0.814840 -0.221835  1.098872      2.326189      0.32
...          ...       ...       ...           ...       ...
420331  0.999312  0.094957  0.855602      1.471563      1.02
420332  0.999115  0.093348  0.831275      1.478324      1.04
420333  0.998231  0.094957  0.843438      1.468057      0.57
420334  0.998231  0.094689  0.843438      1.468308      0.82
420335  0.998231  0.093884  0.879929      1.471563      0.57 """

""" print(trn_csv.info())
 #   Column        Non-Null Count   Dtype  
---  ------        --------------   -----  
 0   p (mbar)      420336 non-null  float64
 1   T (degC)      420336 non-null  float64
 2    rh (%)       420336 non-null  float64
 3   rho (g/m**3)  420336 non-null  float64
 4   wv (m/s)      420336 non-null  float64
dtypes: float64(5)
memory usage: 16.0 MB
None """

""" print(trn_csv.describe())
            p (mbar)       T (degC)         rh (%)   rho (g/m**3)       wv (m/s)
count  420336.000000  420336.000000  420336.000000  420336.000000  420336.000000
mean        0.743064       0.253618      -0.000614      -0.001843       1.702347
std         0.082039       0.225624       1.000109       0.998787      65.463404
min         0.000000      -0.617221      -3.836470      -3.922624   -9999.000000
25%         0.693857       0.090665      -0.654497      -0.715459       0.990000
50%         0.746634       0.253219       0.192691      -0.057149       1.760000
75%         0.797346       0.414968       0.813030       0.664763       2.860000
max         1.000000       1.000000       1.457696       4.443098      28.490000 """

### Custom_Dataset(Dataset)
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

class Custom_Dataset(Dataset):
    def __init__(self, df, timesteps):
        super().__init__()
        self.csv = df
        self.timesteps =  timesteps
        
        self.x = self.csv.iloc[:,:-1].values
        self.y = self.csv.iloc[:,-1].values
        
        self.x = torch.tensor(self.x, dtype=torch.float32).to(DEVICE)
        self.y = torch.tensor(self.y, dtype=torch.float32).to(DEVICE)
    
    def __len__(self):
        return len(self.x) - self.timesteps
    
    def __getitem__(self, index):
        x = self.x[index : index + self.timesteps]
        y = self.y[index + self.timesteps]
        return x, y

DS = Custom_Dataset(trn_csv, 216)

trn_ldr = DataLoader(DS, batch_size=10000)

for batch_idx, (x_bat, y_bat) in enumerate(trn_ldr):
    print(f'🔹🔹🔹🔹🔹 batch : {batch_idx} 🔹🔹🔹🔹🔹')
    print(f'x.shape : {x_bat.shape}')
    print(f'y.shape : {y_bat.shape}')
    break

##################################################
#2. 모델 정의
##################################################
### 모델 클래스 설정
class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.RNN = nn.RNN(input_size=4,
                          hidden_size=64,
                          num_layers=3,
                          batch_first=True)
        
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.rlu = nn.ReLU()
        
    def forward(self, x):
        x, _ = self.RNN(x)
        x = x[:,-1,:]
        x = self.fc1(x)
        x = self.rlu(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x
    
path_save = './Study25/_data/kaggle/jena/save/'

model = RNN().to(DEVICE)

model.load_state_dict(torch.load(path_save + 'jena_practice.pth', map_location=DEVICE))

EPOCHS = 100

##################################################
#3. 컴파일 훈련
##################################################
loss = nn.MSELoss()

##################################################
#4. 평가 예측
##################################################
def EVALUATE(MODL, LOSS, LODR):
    MODL.eval()
    
    tst_lss = 0
    
    for XTST, YTST in LODR:
        with torch.no_grad():
            x_tst_prd = MODL(XTST)
            lss = LOSS(x_tst_prd, YTST)
        
        tst_lss += lss
        
        return tst_lss / len(LODR)

tst_lss = EVALUATE(model, loss, trn_ldr)

print('불러온 성능도 개구리', tst_lss.item())
# 불러온 성능도 개구리 0.0518173910677433
