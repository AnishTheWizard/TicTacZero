import numpy as np

class TicTacToe:
  def __init__(self):
    ...
  
  def is_win(self, board, player) -> bool:
    ...

  def has_legal_moves(self, board) -> bool:
    ...

  def get_player_perspective(self, board, player) -> np.ndarray:
    return player * board

  def get_reward_for_player(self, board, player) -> int:
    ...

  def get_valid_moves(self, board):
    return np.array([[board[i][j] == 0 for j in range(3)] for i in range(3)])

  def create_blank_board(self) -> np.ndarray:
    ...

  def get_next_state(self, board, action, player) -> np.ndarray:
    board = np.copy(board)
    board[action[0]][action[1]] = player
    return board