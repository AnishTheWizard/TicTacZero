import numpy as np
from game import TicTacToe
import math
import random
from torch import FloatTensor

from model import TicTacPolicyNetwork, TicTacValueNetwork


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
  
  def expand(self, game: TicTacToe, p_nn: TicTacPolicyNetwork, v_nn: TicTacValueNetwork):
    action_board: np.ndarray = game.get_valid_moves(self.state)
    model_input = FloatTensor(self.state.astype(np.float64).ravel())
    priors, value = p_nn(model_input), v_nn(model_input)
    priors = priors.detach().numpy() * action_board.flatten()
    sum = np.sum(priors)
    if sum  == 0:
      print("Priors add up to zero:")
      print("Action Board: ", action_board)
      print("Model Input: ", model_input)
      print("Priors: ", priors)
      print("Value: ", value)
      quit()
    priors /= np.sum(priors)

    self.value_sum += value
    for iy, ix in np.ndindex(action_board.shape):
      if priors[iy * 3 + ix] != 0:
        self.children[(iy, ix)] = Node(priors[iy * 3 + ix],
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
  def __init__(self, game, p_n: TicTacPolicyNetwork, v_n: TicTacValueNetwork):
    self.game: TicTacToe = game
    self.p_n: TicTacPolicyNetwork = p_n
    self.v_n: TicTacValueNetwork = v_n

  def run_simulation(self, num_simulations: int, current_state: np.ndarray):
    root: Node = Node(0, current_state, 1)

    root.expand(self.game, self.p_n, self.v_n)

    
    # print("STARTING SIMULATION")
    
    for simulation in range(num_simulations):
      current_node = root
      search_path = [current_node]

      while current_node.expanded():
        current_node = current_node.select_next_state()
        search_path.append(current_node)

      # First check if the game has ended in simulation
      value = self.get_value_if_possible(current_node.state)

      # if the game isn't over, take a guess
      if value is None:
        value = current_node.expand(self.game, self.p_n, self.v_n)

      self.backtrack(search_path, value)

    # Now find the best child of the root node

    truth_probabilities = self.game.create_blank_board()

    for action, node in root.children.items():
      truth_probabilities[action] = node.visits

    truth_probabilities = truth_probabilities / np.sum(truth_probabilities)
    
    visit_counts = np.array([child.visits for child in root.children.values()])
    actions = np.array([action for action in root.children.keys()])

    best_action = actions[np.argmax(visit_counts)]
  
    return best_action, truth_probabilities
        
  def backtrack(self, search_path: list[Node], value):
    for node in search_path:
      node.value_sum += value
      node.visits = 1 + node.visits


  def get_value_if_possible(self, state):
    if self.game.is_win(state):
      return 1
    elif self.game.is_win(-state):
      return -1
    else:
      return None



if __name__ == '__main__':
  game = TicTacToe()
  mcts = MCTS(game, TicTacPolicyNetwork(), TicTacValueNetwork())
  board = game.create_blank_board()
  board[0,0] = 1
  action, truth = mcts.run_simulation(1, board)
  print(action)
