import torch
from sympy import Float

from torch import FloatTensor
from torch.functional import F

from game import TicTacToe
from mcts import MCTS
from model import TicTacModel
import numpy as np


if __name__ == '__main__':
    game = TicTacToe()
    model = TicTacModel()

    model.load_state_dict(torch.load("higher_lr_softmax/model_29.pth"))
    model.eval()

    print(model)

    mcts = MCTS(game, model)

    board = np.array([
        [-1, 1, 0],
        [-1, 0, 0],
        [0, 0, 1]
    ])
    print(board)
    action, truth = mcts.run_simulation(100, board, 2)
    np.set_printoptions(precision=3, suppress=True)
    torch.set_printoptions(precision=3, sci_mode=False)
    #
    print(truth)
    print(action)


    pi, v = model(FloatTensor(board.flatten()))
    pi = F.softmax(pi, dim=-1)
    print(pi.reshape((3,3)))
    print(v)
