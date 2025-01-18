import torch
from torch import nn
from torch.nn import init


class TicTacModel(nn.Module):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.layers = nn.Sequential(
      nn.Linear(3 * 3, 16, True),
      nn.ReLU(False),
      nn.Linear(16, 16, True),
      nn.ReLU(False)
    )

    self.softmax = nn.LogSoftmax(dim=-1)

    self.policy_head = nn.Linear(16, 9, True)
    self.value_head = nn.Linear(16, 1, True)

    self.init_weights()

  def init_weights(self):
    for layer in self.layers:
      if isinstance(layer, nn.Linear):
        init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
          init.zeros_(layer.bias)

    init.xavier_uniform_(self.policy_head.weight)
    if self.policy_head.bias is not None:
      init.zeros_(self.policy_head.bias)

    init.xavier_uniform_(self.value_head.weight)
    if self.value_head.bias is not None:
      init.zeros_(self.value_head.bias)

  def forward(self, x):
    x = self.layers(x)
    return self.softmax(self.policy_head(x)), self._sigmoid(self.value_head(x))

  def _sigmoid(self, x):
    return nn.functional.sigmoid(x) * 2 - 1
# class TicTacPolicyNetwork(nn.Module):
#   def __init__(self, *args, **kwargs):
#     super().__init__(*args, **kwargs)
#
#     self.layers = nn.Sequential(
#       nn.Linear(3 * 3, 64, True),
#       nn.ReLU(False),
#       nn.Linear(64, 32, True),
#       nn.ReLU(False),
#       nn.Linear(32, 9, True)
#     )
#
#   def forward(self, x):
#     x = self.layers(x)
#     return nn.functional.softmax(x, dim=-1)
#
#
#
# class TicTacValueNetwork(nn.Module):
#   def __init__(self, *args, **kwargs):
#     super().__init__(*args, **kwargs)
#
#     self.layers = nn.Sequential(
#       nn.Linear(3 * 3, 64, True),
#       nn.ReLU(False),
#       nn.Linear(64, 32, True),
#       nn.ReLU(False),
#       nn.Linear(32, 1, True)
#     )
#
#   def forward(self, x):
#     x = self.layers(x)
#
#     return nn.functional.sigmoid(x)