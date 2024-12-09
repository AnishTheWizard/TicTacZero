import torch

from torch import FloatTensor
from torch.functional import F

from game import TicTacToe
from mcts import MCTS
from model import TicTacModel
import numpy as np


if __name__ == '__main__':
    game = TicTacToe()
    model = TicTacModel()

    model.load_state_dict(torch.load("models/model_25.pth"))
    model.eval()

    print(model)

    mcts = MCTS(game, model)

    board = np.array([
        [1, -1, 1],
        [-1, 1, -1],
        [0, 0, 0]
    ])

    action, truth = mcts.run_simulation(20, board)

    print(action)