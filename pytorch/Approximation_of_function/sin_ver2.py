import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import matplotlib.pyplot as plt
import numpy as np
import math

# シード値を設定する
torch.manual_seed(1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 3)
        self.fc2 = nn.Linear(3, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        # x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#データ生成を行う
def generateData(batch_size=128, shuffle=False):
    trX = torch.arange(0, 2*math.pi, 0.01) # 0から2piの範囲で、
    trX = torch.reshape(trX, (1, len(trX))) # ベクトルを行列に変換
    trX = trX.t() #len(x)行1列に転置する
    trY = torch.sin(trX)
    train_ = torch.utils.data.TensorDataset(trX, trY)
    
    train_iter = torch.utils.data.DataLoader(train_, batch_size=batch_size, shuffle=shuffle)
    return train_iter

def graph(model):
    trX = torch.arange(0, 2*math.pi, 0.1) # 0から2piの範囲で、
    trX = torch.reshape(trX, (1, len(trX))) # ベクトルを行列に変換
    trX = trX.t() #len(x)行1列に転置する
    trY = torch.sin(trX)

    y = model(trX)

    plt.plot(trX[:,0].numpy(), trY.numpy())
    plt.plot(trX[:,0].numpy(), y.data.numpy())
    plt.show()

def train(model, train_loader, optimizer, epoch, log_interval=10):
    model.train() # ネットワークを学習モードにする
    loss_func = nn.MSELoss(reduction='sum')
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        # イテレーション(log_interval)毎に結果を出力する
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def main(epochs=100, save_model=False):
    #モデル定義
    model = Net()
    optimizer = optim.Adam(model.parameters(),lr=0.1)

    train_loader = generateData(batch_size=64,shuffle=True)
    
    for epoch in range(1, epochs + 1):
        train(model, train_loader, optimizer, epoch)

    graph(model)

if __name__ == '__main__':
    main()