import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.linear = nn.Linear(input, output)  #x is the input, y is the output. expected values is (1,1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred