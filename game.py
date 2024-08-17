import numpy as np

class TicTacToe:
  def __init__(self):
    ...
  
  def is_win(self, board, player) -> bool:
    ...

  def has_legal_moves(self, board) -> bool:
    ...

  def get_player_perspective(self, board, player) -> np.array:
    return player * board

  def get_reward_for_player(self, board, player) -> int:
    ...

  def get_valid_moves(self, board) -> np.array:
    ...

  def create_blank_board(self) -> np.array:
    ...

  def get_next_state(self, board, player, action) -> tuple:
    ...