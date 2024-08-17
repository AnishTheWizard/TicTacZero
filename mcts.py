
class Node:
  def __init__(self):
    self.prior = 0
    self.value_sum = 0
    self.visits = 0
    self.children = {}

  def value(self) -> float:
    return self.value_sum / self.visits
  


class MCTS:
  def __init__(self):
    ...

  def run_simulation(self, num_simulations: int, current_state: Node):
    # Define Root
    root = current_state
    search_path = list()
    # Expand Root
    root.expand()

    # Start simulation
    for iter in range(num_simulations):
      current_node = root
      # Find first unexpanded child of the current node (root for i = 0)
      while current_node.is_expanded():
        # Use UCB to do so
        search_path.append(current_node)
        current_node = current_node.select_node()

      # Expand this node using the priors from the policy network
      priors, value = model(current_node.state)
      current_node.create_children(priors)

      # Use the search path to adjust all affected nodes for simulation run 
      self.backtrack(search_path)