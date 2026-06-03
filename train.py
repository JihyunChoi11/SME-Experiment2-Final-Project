import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

def train_model():
    mat_path = 'DH_FR1.mat'
    data = sio.loadmat(mat_path, squeeze_me=False)
    
    d_hat = np.asarray(data['d_hat'], dtype=float)  
    p_true = np.asarray(data['p'], dtype=float)     
    
    X = d_hat.T
    y = p_true.T
    
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    class AdvancedResidualNet(nn.Module):
        def __init__(self):
            super(AdvancedResidualNet, self).__init__()
            self.input_bn = nn.BatchNorm1d(18)
            self.fc1 = nn.Linear(18, 64)
            self.bn1 = nn.BatchNorm1d(64)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.1)
            self.fc2 = nn.Linear(64, 64)
            self.bn2 = nn.BatchNorm1d(64)
            self.fc_out = nn.Linear(64, 2)
            
        def forward(self, x):
            x_norm = self.input_bn(x)
            out1 = self.relu(self.bn1(self.fc1(x_norm)))
            out1 = self.dropout(out1)
            out2 = self.bn2(self.fc2(out1))
            
            res = self.relu(out1 + out2)
            return self.fc_out(res)
            
    model = AdvancedResidualNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.02, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=300, eta_min=0.0005)
    
    model.train()
    for epoch in range(300):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
    torch.save(model.state_dict(), 'model.pt')
    print("최종 고도화 가중치 파일(model.pt) 추출 완료")

if __name__ == "__main__":
    train_model()