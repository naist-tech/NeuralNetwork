# Code in file nn/two_layer_net_optim.py
import torch
import math
import matplotlib.pyplot as plt

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1, 3, 1

# Create random Tensors to hold inputs and outputs.
x = torch.arange(0, 2*math.pi, 0.1)
x = torch.reshape(x, (1, len(x)))
x = x.t()
y = torch.sin(x)
torch.manual_seed(1)
loss_x  = []
lost_y = []

# Use the nn package to define our model and loss function.
model = torch.nn.Sequential(
          torch.nn.Linear(D_in, H),
          torch.nn.Sigmoid(),
          torch.nn.Linear(H, D_out),
        )
loss_fn = torch.nn.MSELoss(reduction='sum')

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(500):
  for i in range(50):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x)

    # Compute and print loss.
    loss = loss_fn(y_pred, y)
    
    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the Tensors it will update (which are the learnable weights
    # of the model)
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its parameters
    optimizer.step()

  print(t*50+50, loss.item())
  loss_x.append(t*50+50)
  lost_y.append(loss.item())
  if loss.item() < 0.05:
    break
  


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