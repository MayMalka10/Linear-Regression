import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from Model import Linear

x = torch.linspace(0,100,201).reshape(-1,1)                  # 201 evenly spaced points
real_w = 3; real_b = 200                                      # real weight and bias values
torch.manual_seed(49)
noise = torch.randint(-15,15,size=(1,201)).reshape(-1,1)     # noise vector
y = real_w*x + real_b + noise                                # our given noisy linear vector

# PLOT THE Y VECTOR POINTS

plt.scatter(np.array(x),np.array(y))
plt.show()

# Create the model
torch.manual_seed(54)
input_dim = 1
output_dim = 1
model = Linear(input_dim,output_dim)
initial_weight = model.linear.weight.item()
initial_bias = model.linear.bias.item()
initial_y = (initial_weight*x + initial_bias)

# plot the initial linear line guess.

plt.scatter(np.array(x),np.array(y))
plt.plot(x,initial_y,'r')
plt.show()

#adjusting the learning process!

criteria = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr = 0.0002)
epochs = 20000
losses = []

s = range(epochs)
t = 3
## Learn

for epc in range(epochs):
    epc += 1
    y_pred = model.forward(x)
    loss = criteria(y_pred,y)
    losses.append(loss.item())
    if epc <= 10:
        print(f'epoch: {epc:2}  loss: {loss.item():10.8f}  weight: {model.linear.weight.item():10.8f}  \
        bias: {model.linear.bias.item():10.8f}')
    elif epc == 11:
        print('.\n.\n.')
    elif epc >= epochs-10:
        print(f'epoch: {epc:2}  loss: {loss.item():10.8f}  weight: {model.linear.weight.item():10.8f}  \
        bias: {model.linear.bias.item():10.8f}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Show me what you got (I like what you got)

plt.plot(range(epochs), losses)
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.show()

w1,b1 = model.linear.weight.item(), model.linear.bias.item()
y1 = x*w1 + b1
plt.scatter(x.numpy(), y.numpy())
plt.plot(x,y1,'r')
plt.show()