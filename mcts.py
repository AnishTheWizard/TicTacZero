import numpy as np
from game import TicTacToe

class Node:
  def __init__(self, prior, state, player):
    self.prior = prior
    self.state = state
    self.player = player
    self.value_sum = 0
    self.visits = 0
    self.children = {}
  
  def __repr__(self):
    return f"{self.state}"

  def value(self) -> float:
    return self.value_sum / self.visits
  
  def expand(self, game: TicTacToe, model):
    action_board: np.ndarray = game.get_valid_moves(self.state)
    priors, _ = model(self.state)
    print("PRIOR PROB:", priors)
    for iy, ix in np.ndindex(action_board.shape):
      self.children[(iy, ix)] = Node(priors[iy * 3 + ix], 
                                   game.get_next_state(self.state, (iy, ix), -self.player), -self.player)
  


class MCTS:
  def __init__(self, game, model):
    self.game = game
    self.model = model

  def run_simulation(self, num_simulations: int, current_state: np.ndarray):
    root: Node = Node(0, current_state, -1) # put constructor params in

    root.expand(self.game, self.model)

    print("STATE:", root.state, '\nNODE CHILDREN:', root.children)
    

if __name__ == "__main__":
  def FakeModel(board):
    return np.array([1/9 for i in range(9)]), 1
  
  
  mcts = MCTS(TicTacToe(), FakeModel)
  
  mcts.run_simulation(1, np.zeros((3, 3)))