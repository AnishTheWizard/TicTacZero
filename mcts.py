import numpy as np
from game import TicTacToe
import math

# understand this code further
def ucb(parent, child):
  prior_score = child.prior * np.sqrt(parent.visits) / (child.visits + 1)
  
  if child.visits > 0:
    value_score = -child.value()
  else:
    value_score = 0
  
  return value_score + prior_score

class Node:
  def __init__(self, prior, state, player):
    self.prior = prior
    self.state = state
    self.player = player
    self.value_sum = 0
    self.visits = 0
    self.children = {}
  
  def __repr__(self):
    return f"\n{self.state}, {(self.value_sum, self.visits)}, {self.prior}\n"

  def value(self) -> float:
    return self.value_sum / self.visits if self.visits > 0 else 0
  
  def expand(self, game: TicTacToe, model):
    action_board: np.ndarray = game.get_valid_moves(self.state)
    priors, value = model(self.state)
    self.value_sum += value
    print("PRIOR PROB:", priors)
    for iy, ix in np.ndindex(action_board.shape):
      self.children[(iy, ix)] = Node(priors[iy * 3 + ix], #should multiply this by action board's value to make sure the prior isn't there for impossible moves
                                   game.get_next_state(self.state, (iy, ix), -self.player), -self.player)
      
    return value
  
  def expanded(self):
    return len(self.children) > 0
  
  def select_next_state(self):
    maxUCB = 0
    maxNode = self.children[list(self.children.keys())[0]]
    print(self.children)
    for action, node in self.children.items():
      print(node)
      score = ucb(self, node)
      if score > maxUCB:
        maxUCB = score
        maxNode = node
    return maxNode
  


class MCTS:
  def __init__(self, game, model):
    self.game = game
    self.model = model

  def run_simulation(self, num_simulations: int, current_state: np.ndarray):
    root: Node = Node(0, current_state, -1)

    root.expand(self.game, self.model)

    
    print("STARTING SIMULATION:")
    
    for simulation in range(num_simulations):
      
      current_node = root
      search_path = [current_node]
      
      while current_node.expanded():
        current_node = current_node.select_next_state()
        search_path.append(current_node)
        
      pred_value = current_node.expand(self.game, self.model)
      
      self.backtrack(search_path, pred_value)
      
    
    bestNode = root
    bestNodeValue = 0
      
    for action, node in root.children.items():
      print(node)
      if node.value() > bestNodeValue:
        
        bestNodeValue = node.value()
        bestNode = node
      
    # print(bestNode)
  
  def backtrack(self, search_path: list[Node], value):
    for node in search_path:
      node.value_sum += value
      node.visits = 1 + node.visits
    

if __name__ == "__main__":
  def FakeModel(board):
    return np.array([1/9 for i in range(9)]), 1
  
  
  mcts = MCTS(TicTacToe(), FakeModel)
  
  mcts.run_simulation(20, np.zeros((3, 3)))