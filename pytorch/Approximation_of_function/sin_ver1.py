import torch
import math
import matplotlib.pyplot as plt

# D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
D_in, H, D_out = 1, 3, 1
step = 0.1

# シード値を設定する
torch.manual_seed(1)

# 学習データをを作成する
x = torch.arange(0, 2*math.pi, step) # 0から2piの範囲で、
x = torch.reshape(x, (1, len(x))) # ベクトルを行列に変換
x = x.t() #len(x)行1列に転置する
y = torch.sin(x) + torch.sin(2*x)

# 損失を可視化
loss_x  = []
lost_y = []

# モデルの構築と損失関数の設定
model = torch.nn.Sequential(
          torch.nn.Linear(D_in, H),
          torch.nn.Sigmoid(),
          torch.nn.Linear(H, D_out),
        )
loss_fn = torch.nn.MSELoss(reduction='sum')

#学習率を設定し、最適化アルゴリズムをAdamを選択
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 学習
for t in range(500):
  for i in range(50):
    # 順伝搬
    y_pred = model(x)

    # 損失を算出
    loss = loss_fn(y_pred, y)
    
    # 勾配の初期化
    optimizer.zero_grad()

    # 勾配を算出
    loss.backward()

    # パラメータを更新
    optimizer.step()

  # 学習の進捗を出力
  print(t*50+50, loss.item())
  # 損失を記録する
  loss_x.append(t*50+50)
  lost_y.append(loss.item())
  
  #損失が0.05以下なら学習を終了
  if loss.item() < 0.05:
    break
  


# 以下、グラフ表示
import numpy as np
xx = np.arange(0, 2*math.pi, 0.1)
yy = np.sin(xx)

function = model(x).data

plt.subplot(1, 2, 1)
plt.plot(xx, yy)
plt.plot(x.numpy(), function.numpy())
plt.subplot(1, 2, 2)
plt.plot(loss_x, lost_y)
plt.show()