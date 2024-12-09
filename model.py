import torch
from torch import nn

# class TicTacModel(nn.Module):
#   def __init__(self, *args, **kwargs):
#     super().__init__(*args, **kwargs)
#
#     self.layers = nn.Sequential(
#       nn.Linear(3 * 3, 64, True),
#       nn.ReLU(False),
#       nn.Linear(64, 32, True),
#       nn.ReLU(False),
#
#     )
#
#     self.policy_head = nn.Linear(32, 9, True)
#     self.value_head = nn.Linear(32, 1, True)
#
#   def forward(self, x):
#     x = self.layers(x)
#     return nn.functional.relu(self.policy_head(x)), nn.functional.sigmoid(self.value_head(x))



class TicTacPolicyNetwork(nn.Module):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.layers = nn.Sequential(
      nn.Linear(3 * 3, 64, True),
      nn.ReLU(False),
      nn.Linear(64, 32, True),
      nn.ReLU(False),
      nn.Linear(32, 9, True)
    )

  def forward(self, x):
    x = self.layers(x)
    return nn.functional.softmax(x, dim=-1)



class TicTacValueNetwork(nn.Module):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.layers = nn.Sequential(
      nn.Linear(3 * 3, 64, True),
      nn.ReLU(False),
      nn.Linear(64, 32, True),
      nn.ReLU(False),
      nn.Linear(32, 1, True)
    )

  def forward(self, x):
    x = self.layers(x)

    return nn.functional.sigmoid(x)