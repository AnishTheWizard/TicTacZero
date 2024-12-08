import numpy as np
from game import TicTacToe
import math
import random
from torch import FloatTensor

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
    priors, value = model(FloatTensor(self.state.astype(np.float64).ravel()))
    self.value_sum += value
    for iy, ix in np.ndindex(action_board.shape):
        self.children[(iy, ix)] = Node(priors[iy * 3 + ix], #should multiply this by action board's value to make sure the prior isn't there for impossible moves
                                   game.get_next_state(self.state, (iy, ix)), -self.player)
    return value
  
  def expanded(self):
    return len(self.children) > 0
  
  def select_next_state(self):
    maxUCB = 0 # shouldn't we find the ucb of the maxNode?
    maxNode = self.children[list(self.children.keys())[0]]
    for action, node in self.children.items():
      score = ucb(self, node)
      if score > maxUCB:
        maxUCB = score
        maxNode = node
    return maxNode
  


class MCTS:
  def __init__(self, game, model):
    self.game: TicTacToe = game
    self.model = model

  def run_simulation(self, num_simulations: int, current_state: np.ndarray):
    root: Node = Node(0, current_state, -1)

    root.expand(self.game, self.model)

    
    # print("STARTING SIMULATION")
    
    for simulation in range(num_simulations):
      current_node = root
      search_path = [current_node]
      
      while current_node.expanded():
        current_node = current_node.select_next_state()
        search_path.append(current_node)
        
      pred_value = current_node.expand(self.game, self.model)
      
      self.backtrack(search_path, pred_value)
      
    
    bestAction, bestNodeValue = list(root.children.items())[0]
    bestNodeValue = bestNodeValue.value()
    truth_probabilities = self.game.create_blank_board()
    
    valid_actions = self.game.get_valid_moves(current_state)

    vectorized_valid_actions = list()
      
    for action, node in root.children.items():
      if valid_actions[action]:
        vectorized_valid_actions.append(action)
        truth_probabilities[action] = (node.visits / root.visits) if node.visits > 0 and root.visits > 0 else 0
      else:
        truth_probabilities[action] = 0
      
    if sum := np.sum(truth_probabilities) > 0:
      truth_probabilities = truth_probabilities / sum
      
    validated_truth = [float(truth_probabilities[i]) for i in vectorized_valid_actions]
    # print(vectorized_valid_actions, validated_truth)
    try:
      bestAction = vectorized_valid_actions[np.random.choice(range(len(vectorized_valid_actions)), p=validated_truth)]
    except:
      flat_index = np.argmax(validated_truth)
      bestAction = vectorized_valid_actions[flat_index]
  
    return bestAction, truth_probabilities
      
        
  def backtrack(self, search_path: list[Node], value):
    for node in search_path:
      node.value_sum += value
      node.visits = 1 + node.visits