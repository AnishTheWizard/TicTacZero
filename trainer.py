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
      examples[i][2] = 1 * ((-1) ** i)
      # print('-------------------------------')
      # print(examples[i])
    return examples

  def find_loss_pi(self, outputs, targets):
    loss = -(targets * torch.log(outputs)).sum(dim=1)
    return loss.mean()
  
  def execute_episode(self, game: TicTacToe, model: TicTacModel):
    # execute a game using the MCTS (let the model predict on its own)
    board = game.create_blank_board()
    examples = list()
    mcts = MCTS(game, model)
    iteration = 0
    while True:
      
      iteration += 1
      
      action, truth = mcts.run_simulation(20, board)
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
    
  def train_model(self, examples, model: TicTacModel):
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    # value_optimizer = torch.optim.Adam(v_n.parameters(), lr=0.001)
    criterion_policy = torch.nn.MSELoss()
    criterion_value = torch.nn.MSELoss()

    boards, y = Tensor([e[0].ravel() for e in examples]), Tensor(list(range(len(examples))))
    dataset = TensorDataset(boards, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    pi_losses = list()
    v_losses = list()
    epochs = 50

    for i in range(1, epochs + 1):
      model.train()
      for batch_x, batch_y in dataloader:

        boards = batch_x
        pis = np.array([examples[int(i)][1].ravel() for i in batch_y]).astype(np.float64)
        vs = np.array([examples[int(i)][2] for i in batch_y]).astype(np.float64)
        target_pis = torch.FloatTensor(pis)
        target_vs = torch.FloatTensor(vs)

        boards = boards.contiguous()
        target_pis = target_pis.contiguous()
        target_vs = target_vs.contiguous()

        output_pi, output_v = model(boards)

        output_pi = F.softmax(output_pi, dim=1)

        loss_pi = self.find_loss_pi(output_pi, target_pis)
        loss_v = criterion_value(output_v.view(-1), target_vs)
        total_loss = loss_pi + loss_v

        pi_losses.append(float(loss_pi))
        v_losses.append(float(loss_v))

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

      if i % 25 == 0:
        torch.set_printoptions(precision=4, sci_mode=False)
        print(f"-------------------Epoch {i}-------------------")
        print("Policy Loss", np.mean(pi_losses))
        print("Value Loss", np.mean(v_losses))
        print("Examples:")
        print(boards[0])
        print(output_pi[0].detach())
        print(target_pis[0])

    return model
    
  def learn(self):
    model = TicTacModel()
    game = TicTacToe()
    
    num_eps = 50 #50
    num_training_sets = 30 #100


    for i in range(num_training_sets):
      examples = list()
      for e in range(num_eps):
        examples.extend(self.execute_episode(game, model))
      print(f"Training with {len(examples)} examples: model_{i}")
      model = self.train_model(examples, model)
      torch.save(model.state_dict(), f"models/model_{i}.pth")
    return model
    
  
  
  
if __name__ == "__main__":
  trainer = Trainer()

  trainer.learn()