import numpy as np

class TicTacToe:
  def __init__(self):
    ...
  
  def is_win(self, board: np.ndarray) -> bool: # assume in perspective of player (1) 
    identity_vector = np.array([1,1,1])
    row_based_win = 3 in board.dot(identity_vector)
    col_based_win = 3 in board.T.dot(identity_vector)
    diag_based_win = board.trace() == 3 or np.flipud(board).trace() == 3
    
    return row_based_win or col_based_win or diag_based_win or (0 not in board)
  

  def has_legal_moves(self, board) -> bool:
    ...

  def get_player_perspective(self, board, player) -> np.ndarray:
    return player * board

  def get_reward_for_player(self, board, player) -> int:
    ...

  def get_valid_moves(self, board):
    return np.array([[board[i][j] == 0 for j in range(3)] for i in range(3)])

  def create_blank_board(self) -> np.ndarray:
    return np.zeros((3, 3))

  def get_next_state(self, board, action) -> np.ndarray: # assume in perspective of player 1
    board = np.copy(board)
    board[action[0]][action[1]] = 1
    return board