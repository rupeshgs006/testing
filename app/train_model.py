from sklearn.linear_model import LinearRegression

import joblib
import torch
import torch.nn as nn
import torch.optim as optimizer
import numpy as np

# X = np.array([[1], [2], [3], [4], [5]])
# y = np.array([2, 4, 6, 8, 10])

# model = LinearRegression()
# model.fit(X,y)

# joblib.dump(model,"model.joblib")
# print("model saved")

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,1)

    def forward(self,x):
        return self.linear(x)
    

x = torch.tensor([[1.], [2.], [3.], [4.], [5.]])
y = torch.tensor([[2.], [4.], [6.], [8.], [10.]])

model = LinearModel()
loss_fn = nn.MSELoss()
opt = optimizer.SGD(model.parameters(),lr=0.01)

for _ in range(1000):
    preds = model(x)
    loss = loss_fn(preds,y)

    opt.zero_grad()
    loss.backward()
    opt.step()

torch.save(model.state_dict(),"torch_model.pt")
print("model saved")