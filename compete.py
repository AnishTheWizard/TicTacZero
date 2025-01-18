import torch


from game import TicTacToe
from model import TicTacModel
import numpy as np
from minimax import minimax
from mcts import MCTS


if __name__ == '__main__':
    game = TicTacToe()
    model = TicTacModel()



    # play a game



    for model_num in [13]:
        model.load_state_dict(torch.load(f"simulated_annealing/model_{model_num}.pth"))
        model.eval()

        mcts = MCTS(game, model)
        wins = list()
        for i in range(100):
            print(f"------------Game {i}------------")
            board = game.create_blank_board()
            playerOnePlaying = True
            while True:
                if playerOnePlaying:
                    action, truth = mcts.run_simulation(100, board, 1)
                    score = truth[action[0]][action[1]]
                else:
                    action, score = minimax(game, board, 0, True, 0)

                board = game.get_next_state(board, action)
                # print(f"--------------Player {1 if playerOnePlaying else 2} played {action} with score {score}--------------")
                # print(board)
                if 0 not in board:
                    print("Game Tied!")
                    print(board)
                    wins.append(True)
                    break
                elif game.is_win(board):
                    # print(f"Player {1 if playerOnePlaying else 2} wins!")
                    wins.append(playerOnePlaying)
                    if playerOnePlaying:
                        print("Player 1 won!")
                        print(board)
                    break
                else:
                    board *= -1
                    playerOnePlaying = not playerOnePlaying

        wins = np.array(wins)
        print(f"Win Rate for Model_{model_num}:", len(wins[wins == 1]) / len(wins))