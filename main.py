import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn

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
model.load_state_dict(torch.load('model.pt', map_location=torch.device('cpu')))
model.eval()

def your_algorithm(d_hat_u, p_bs):
    """
    채점기 루프에 대응하여 u번째 사용자의 18차원 거리를 기반으로 2차원 좌표 추정
    """
    # 1차원 수치 배열 d_hat_u (18,)를 파이토치 행렬 규격인 (1, 18) 차원으로 변환
    inputs = torch.FloatTensor(d_hat_u).unsqueeze(0)
    
    # 미분 연산 메모리 할당을 원천 차단하여 추론 속도 극대화 (10분 제한 방어 핵심)
    with torch.no_grad():
        prediction = model(inputs).numpy()
        
    return prediction[0]

def main():
    # 1) 입력 데이터 로드 — 채점기가 같은 폴더에 .mat 파일 자동 배치
    mat_path = 'DH_FR1.mat'
    
    data = sio.loadmat(mat_path, squeeze_me=False)
    BS_positions = np.asarray(data['BS_positions'] if 'BS_positions' in data else data['p_bs'], dtype=float)
    d_hat = np.asarray(data['d_hat'], dtype=float)    # (18, num_user)

    # 2) 본인 알고리즘 — 사용자 수는 입력에서 동적으로 받기
    num_user = d_hat.shape[1]
    p_hat = np.zeros((2, num_user))
    
    for u in range(num_user):
        p_hat[:, u] = your_algorithm(d_hat[:, u], BS_positions)

    # 3) 결과 반환 — numpy 배열, 모양 (2, num_user)
    return p_hat

if __name__ == "__main__":
    main()