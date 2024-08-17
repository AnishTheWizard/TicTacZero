import torch
from torch import nn

class TicTacModel(nn.Module):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.layers = nn.Sequential(
      nn.Linear(3 * 3, 16, True),
      nn.ReLU(False),
      nn.Linear(16, 16, True),
      nn.ReLU(False)
    )

    self.action_head = nn.Linear(16, 9, True)
    self.value_head = nn.Linear(16, 1, True)


  def forward(self, x):
    x = self.layers(x)
    return (nn.functional.relu(self.action_head(x)), nn.functional.relu(self.value_head(x)))