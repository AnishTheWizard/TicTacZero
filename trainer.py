from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader

from mcts import MCTS
from model import TicTacModel
from game import TicTacToe
import numpy as np
import random

import torch
from torch.functional import F


class Trainer:
  def __init__(self):
    ...
    
  def assign_rewards(self, examples: list) -> list: # assume in perspective of player 1
    examples.reverse()
    for i in range(len(examples)):
      examples[i][2] = -1 * ((-1) ** i)
      # print('-------------------------------')
      # print(examples[i])
    return examples
  
  def loss_pi(self, outputs, targets):
        loss = -(targets * torch.log(outputs)).sum(dim=1)
        return loss.mean()

  def loss_v(self, outputs, targets):
      loss = torch.sum((targets-outputs.view(-1))**2)/targets.size()[0]
      return loss
  
  
  def execute_episode(self, game: TicTacToe, model: TicTacModel):
    # execute a game using the MCTS (let the model predict on its own)
    board = game.create_blank_board()
    examples = list()
    mcts = MCTS(game, model)
    iteration = 0
    while True:
      
      iteration += 1
      
      action, truth = mcts.run_simulation(100, board)
      # print(f"ITER: {iteration}\n", board, '\n', game.get_valid_moves(board), '\n')
      # print(action, '\n', truth, '\n----------------------\n')
      examples.append([board, truth, None]) # The third entry is for the value state
      board = game.get_next_state(board, action)
      if game.is_win(board):
        # print("GAME OVER:\n", board)
        examples = self.assign_rewards(examples)
        return examples
      board = board * -1 # flips it to the proponent side
    return []
    
  def pit(self, old_model, new_model):
    return 1
    
  def train_model(self, examples, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion_policy = torch.nn.NLLLoss2d()
    criterion_value = torch.nn.MSELoss()

    boards, y = Tensor([e[0].ravel() for e in examples]), Tensor(list(range(len(examples))))
    dataset = TensorDataset(boards, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model.train()


    # while batch_idx < len(examples):
    i = 0
    for batch_x, batch_y in dataloader:
      optimizer.zero_grad()
      boards = batch_x
      pis = np.array([examples[int(i)][1].ravel() for i in batch_y]).astype(np.float64)
      vs = np.array([examples[int(i)][2] for i in batch_y]).astype(np.float64)
      target_pis = torch.FloatTensor(pis)
      target_vs = torch.FloatTensor(vs)

      boards = boards.contiguous()
      target_pis = target_pis.contiguous()
      target_vs = target_vs.contiguous()

      output_pi, output_v = model(boards)

      loss_pi = F.nll_loss(F.log_softmax(output_pi, dim=1), target_pis.argmax(dim=1))
      loss_v = criterion_value(output_v.view(-1), target_vs)
      total_loss = loss_pi + loss_v

      print("IT:", i)
      print("Loss:", loss_pi, loss_v)
      i+=1

      total_loss.backward()
      optimizer.step()

    model.eval()
    return model
    
  def learn(self):
    model = TicTacModel()
    game = TicTacToe()
    
    num_eps = 50
    num_training_sets = 100
    
    
    
    for i in range(num_training_sets):
      examples = list()
      for e in range(num_eps):
        examples.extend(self.execute_episode(game, model))
      print(f"Training with {len(examples)} examples")
      model = self.train_model(examples, model)
    return model
    
  
  
  
if __name__ == "__main__":
  trainer = Trainer()

  trainer.learn()