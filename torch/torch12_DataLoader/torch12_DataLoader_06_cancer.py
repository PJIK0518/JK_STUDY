# torch.11_6.copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import warnings

# warnings.filterwarnings('ignore')

### 랜덤고정
RS = 55
torch.cuda.manual_seed(RS)
torch.manual_seed(RS)
np.random.seed(RS)
random.seed(RS)

### GPU 설정
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'CPU')
print('torch :', torch.__version__)
print('dvice :', DEVICE)

############################################################
#1. 데이터
############################################################
DS = load_breast_cancer()
x = DS.data
y = DS.target

x_trn, x_tst, y_trn, y_tst = train_test_split(
    x, y,
    train_size=0.7,
    shuffle=True,
    random_state=RS
)

Scaler = StandardScaler()
x_trn = Scaler.fit_transform(x_trn)
x_tst = Scaler.transform(x_tst)

x_trn = torch.tensor(x_trn, dtype=torch.float32).to(DEVICE)
x_tst = torch.tensor(x_tst, dtype=torch.float32).to(DEVICE)
y_trn = torch.tensor(y_trn, dtype=torch.float32).unsqueeze(1).to(DEVICE)
y_tst = torch.tensor(y_tst, dtype=torch.float32).unsqueeze(1).to(DEVICE)

### Dataloader
from torch.utils.data import TensorDataset # 데이터 합치기
from torch.utils.data import DataLoader    # Batch 나누기

### 1. x + y dataset
trn_set = TensorDataset(x_trn, y_trn)
tst_set = TensorDataset(x_tst, y_tst)

""" print(trn_set) <torch.utils.data.dataset.TensorDataset object at 0x79ba7f69a890> """
""" print(tst_set) <torch.utils.data.dataset.TensorDataset object at 0x79b9436f0130> """
""" print(type(trn_set)) <class 'torch.utils.data.dataset.TensorDataset'> """
""" print(len(trn_set)) 398 """

""" print(trn_set[0]) : 첫번째 행의 x, y 값
(tensor([ 1.8519,  1.8124,  1.8472,  1.7873,  0.2356,  1.2470,  1.0031,  2.1285,
         0.2864, -0.3585,  2.2273,  1.3129,  2.2919,  1.5146,  0.5369,  1.1569,
         0.9041,  2.2043,  2.2506,  0.1661,  1.7285,  1.3083,  1.6524,  1.4546,
        -0.1764,  0.5694,  0.4967,  1.5909,  0.6430, -0.4063], device='cuda:0'), tensor([0.], device='cuda:0')) """
""" print(trn_set[0][0]) : 첫번째 행의 x 값
tensor([ 1.8519,  1.8124,  1.8472,  1.7873,  0.2356,  1.2470,  1.0031,  2.1285,
         0.2864, -0.3585,  2.2273,  1.3129,  2.2919,  1.5146,  0.5369,  1.1569,
         0.9041,  2.2043,  2.2506,  0.1661,  1.7285,  1.3083,  1.6524,  1.4546,
        -0.1764,  0.5694,  0.4967,  1.5909,  0.6430, -0.4063], device='cuda:0') """
""" print(trn_set[0][1]) : 첫번째 행의 y 값
tensor([0.], device='cuda:0') """

### 2. batch 정의
trn_loader = DataLoader(trn_set, batch_size=100, shuffle=True)
tst_loader = DataLoader(tst_set, batch_size=100, shuffle=False)
""" print(len(trn_loader)) 40 : 398을 10개씩 > 10*39 + 8*1 """
""" print(type(trn_loader)) <class 'torch.utils.data.dataloader.DataLoader'> """

### 3. batch 확인하기
#1) 반복문으로 확인하기
# for x, y in trn_loader:
#     print(x)
#     print(y)
#     break
""" 0 번째 배치
[tensor([[-9.7395e-02, -8.5550e-01, -1.4980e-01, -1.8942e-01, -1.0507e-01,
         -6.8898e-01, -5.0611e-01, -2.6430e-01, -2.9808e-01, -8.5820e-01,
         -2.6425e-01, -1.0267e+00, -4.1756e-01, -2.6896e-01, -1.8007e-01,
         -6.9274e-01, -4.5598e-01, -1.1189e-01, -1.5373e-01, -7.0022e-01,
         -2.6675e-01, -1.0600e+00, -3.6787e-01, -3.1982e-01, -1.7638e-01,
         -7.7389e-01, -5.6356e-01, -3.1982e-01, -5.7507e-01, -1.0063e+00],
        [ 2.6513e-01,  9.5243e-01,  4.1153e-01,  9.8965e-02,  6.1378e-01,
          1.8455e+00,  1.0754e+00,  1.1165e+00, -2.2410e-01,  5.4648e-01,
         -4.3471e-01, -4.5449e-01,  4.2453e-01, -3.1541e-01,  4.0655e-01,
          1.3988e+00,  7.8687e-01,  1.1488e+00,  7.0478e-01,  1.7511e-01,
         -3.9971e-03,  3.4076e-01,  3.8304e-01, -1.3781e-01, -1.3826e-02,
          1.3482e+00,  9.2957e-01,  9.3056e-01,  1.9751e-01,  1.6532e-01],
        [-6.7386e-01, -2.1620e-02, -7.1243e-01, -6.5936e-01, -5.3553e-01,
         -9.5398e-01, -9.0866e-01, -9.5727e-01,  5.1199e-01, -2.0860e-01,
         -8.9144e-01,  4.3473e-01, -8.8585e-01, -6.4033e-01, -3.2791e-01,
         -8.7468e-01, -7.1113e-01, -6.6014e-01,  4.4471e-01, -7.4645e-01,
         -7.4527e-01,  1.4340e-01, -7.7230e-01, -6.7970e-01, -4.1165e-01,
         -9.0284e-01, -9.2585e-01, -8.6064e-01,  9.9804e-01, -7.6676e-01],
        [ 7.4951e-02,  9.3104e-01,  1.5156e-01, -8.5589e-02,  8.3290e-01,
          2.6788e+00,  1.4415e+00,  5.1023e-01,  2.1691e+00,  1.7541e+00,
         -4.1025e-01,  1.7273e+00, -3.5632e-01, -1.6322e-01,  3.6613e-01,
          5.9435e+00,  2.8164e+00,  7.9810e-01,  4.1123e+00,  2.7503e+00,
         -7.6629e-02,  1.8491e+00,  8.9636e-03, -1.7788e-01,  9.1444e-01,
          4.3060e+00,  2.8346e+00,  1.0005e+00,  3.7580e+00,  3.2450e+00],
        [-2.6080e-02, -7.2246e-01, -1.0712e-01, -1.2550e-01, -1.7993e+00,
         -1.0352e+00, -8.4909e-01, -1.0370e+00, -7.0497e-01, -1.1164e+00,
         -5.3217e-01, -1.1015e+00, -5.6908e-01, -3.6396e-01, -1.1785e+00,
         -6.0258e-01, -3.9685e-01, -7.6068e-01, -4.8678e-01, -6.3903e-01,
         -1.2576e-01, -9.1241e-01, -1.9770e-01, -2.0860e-01, -2.0141e+00,
         -8.1055e-01, -7.7952e-01, -1.0477e+00, -5.5534e-01, -9.4591e-01],
        [-5.8766e-02,  7.3386e-01, -9.6842e-03, -1.4861e-01,  1.7164e+00,
          4.7338e-01,  6.7306e-01,  4.8554e-01,  1.3470e-01,  8.3103e-01,
          8.6287e-01,  1.7311e+00,  1.1188e+00,  2.5646e-01,  2.4890e+00,
          4.4101e-01,  4.9252e-01,  1.4349e+00,  8.9585e-01,  1.7302e-01,
          6.2226e-02,  1.3404e+00,  1.6840e-01, -9.7924e-02,  1.7871e+00,
          3.8356e-01,  5.4847e-01,  8.4666e-01,  3.2902e-01,  5.0328e-01],
        [ 2.9781e-01, -5.3478e-01,  2.3132e-01,  1.4248e-01,  2.0028e-01,
         -4.4747e-01, -7.7727e-01, -2.5089e-01, -6.5048e-02, -8.7208e-01,
         -4.6643e-01,  2.9848e-02, -4.7194e-01, -3.3688e-01, -6.8622e-01,
         -6.2687e-01, -8.1558e-01, -4.2820e-01, -8.8087e-01, -7.3081e-01,
          2.4116e-03, -4.2623e-01, -4.6228e-02, -1.2164e-01, -4.4160e-01,
         -5.5517e-01, -9.8775e-01, -4.3432e-01, -7.8218e-01, -1.0502e+00],
        [-7.4221e-01, -7.5334e-01, -7.6891e-01, -7.0737e-01, -3.5034e-01,
         -8.3216e-01, -7.7921e-01, -6.4231e-01, -6.0509e-01,  1.3007e-01,
         -8.5093e-01, -8.3599e-01, -8.9588e-01, -6.1769e-01,  8.1074e-01,
         -9.0502e-01, -7.4502e-01, -8.7398e-01, -3.4347e-01, -2.7785e-01,
         -7.6663e-01, -7.8084e-01, -8.1063e-01, -6.9281e-01,  9.0588e-01,
         -8.8198e-01, -8.3263e-01, -7.3526e-01, -4.7410e-02,  1.9847e-01],
        [-3.2026e-01, -1.4447e+00, -3.8649e-01, -3.6767e-01, -1.8495e+00,
         -1.2473e+00, -8.2137e-01, -9.5110e-01, -1.7037e+00, -9.6785e-01,
         -9.1782e-01, -1.3920e+00, -8.8269e-01, -6.0765e-01, -8.1330e-01,
         -6.4037e-01, -5.0847e-01, -9.4025e-01, -5.2393e-01, -6.0045e-01,
         -5.2951e-01, -1.6232e+00, -5.7361e-01, -5.1099e-01, -1.5299e+00,
         -8.6491e-01, -7.3310e-01, -9.1890e-01, -9.3834e-01, -7.8976e-01],
        [-8.6404e-01, -1.0384e+00, -8.8574e-01, -7.8089e-01, -1.2940e+00,
         -1.0473e+00, -7.6309e-01, -1.0709e+00, -9.7129e-01, -8.9235e-02,
         -7.9436e-01, -9.7552e-01, -8.0455e-01, -6.0065e-01, -9.8517e-01,
         -5.7342e-01, -4.1279e-01, -1.2210e+00, -1.0918e+00, -9.4303e-02,
         -7.3031e-01, -8.5144e-01, -7.7628e-01, -6.5526e-01, -8.5226e-01,
         -4.0219e-01, -2.9679e-01, -8.8938e-01, -4.7644e-01,  4.0970e-01]],
       device='cuda:0'), tensor([[1.],
        [0.],
        [1.],
        [0.],
        [1.],
        [0.],
        [1.],
        [1.],
        [1.],
        [1.]], device='cuda:0')] """

#2) next() 사용
# iterater는 아니라서 우선 iter 화 시켜야함
iter_trn = iter(trn_loader)
btch_trn = next(iter_trn)

print('🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹')

############################################################
#2. 모델
############################################################
class Model(nn.Module):
    def __init__(self, ID, OD):
        super().__init__()
        self.lin1 = nn.Linear(ID, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, 32)
        self.lin4 = nn.Linear(32, 16)
        self.lin5 = nn.Linear(16, OD)
        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()
        
    def forward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.lin3(x)
        x = self.relu(x)
        x = self.lin4(x)
        x = self.lin5(x)
        x = self.sigm(x)
        
        return x

model = Model(30, 1).to(DEVICE)

EPOCH = 1000

############################################################
#3. 컴파일 훈련
############################################################
loss = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def TRAIN(MODL, LOSS, OPTM, trn_loader): # epo 단위의 훈련 > batch 단위의 훈련
    MODL.train()
    
    total_loss = 0
    
    for XTRN, YTRN in trn_loader:
        OPTM.zero_grad()
        
        x_trn_prd = MODL(XTRN)
        trn_loss = LOSS(x_trn_prd, YTRN)
        
        trn_loss.backward()
        OPTM.step() 
        total_loss = total_loss + trn_loss.item()
        total_loss += trn_loss.item()
        
    return total_loss / len(trn_loader)

for e in range(1, EPOCH+1):
    trn_loss = TRAIN(model, loss, optimizer, trn_loader)
    print('epo :', e)           # verbose 를 직접 설정
    print('bce :', trn_loss)
    
print('🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹')

############################################################
#4. 평가 예측
############################################################
def EVALUATE(MODL, LOSS, tst_loader):
    MODL.eval()
    total_loss = 0
    
    for XTST, YTST in tst_loader:
        with torch.no_grad():
            x_tst_prd = MODL(XTST)
            tst_loss = LOSS(x_tst_prd, YTST)
            total_loss += tst_loss.item()     
               
    return total_loss / len(tst_loader)

tst_loss = EVALUATE(model, loss, tst_loader)

print('bce :', tst_loss)

x_tst_prd = model(x_tst)

# print(type(x_tst_prd)) <class 'torch.Tensor'> 
# :: x_tst_prd 및 y_tst는 torch.Tensor 형태 
# :: accuracy_score는 numpy 형태로 cpu에서 계산 

y_tst = y_tst.detach().cpu().numpy()
x_tst_prd = np.round(x_tst_prd.detach().cpu().numpy())

acc = accuracy_score(y_tst, x_tst_prd)

print('acc :', acc)

# bce : 0.7100207209587097
# acc : 0.9707602339181286