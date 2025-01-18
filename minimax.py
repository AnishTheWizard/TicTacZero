from game import TicTacToe
import numpy as np

def score(game: TicTacToe, board, recursive_depth: int):
    if game.is_win(board):
        return 10 - recursive_depth
    elif game.is_win(-board):
        return -10 + recursive_depth
    else:
        return 0

def minimax(game: TicTacToe, board, recursive_depth: int, is_player: bool, temperature: int = 0):
    if game.is_win(board) or game.is_win(-board):
        return score(game, board, recursive_depth)

    scores = []
    actions = []


    valid_actions = game.get_valid_moves(board)
    valid_actions = np.argwhere(valid_actions == 1)

    for action in valid_actions:
        potential_state = game.get_next_state(board, action)

        scores.append(minimax(game, potential_state, recursive_depth+1, not is_player))
        actions.append(action)

    if temperature != 0:
        score_dist = np.exp(np.array(scores) / temperature)
        score_dist /= np.sum(score_dist)

        if is_player:
            max_score_idx = np.random.choice(range(len(scores)), p=score_dist)
            if recursive_depth == 0:
                return actions[max_score_idx], scores[max_score_idx]

        else:
            min_score_idx = np.random.choice(range(len(scores)), p=(1-score_dist))
            return scores[min_score_idx]

    if is_player:
        max_score_idx = np.argmax(scores)
        if recursive_depth == 0:
            return actions[max_score_idx], scores[max_score_idx]
        return scores[max_score_idx]
    else:
        min_score_idx = np.argmin(scores)
        return scores[min_score_idx]
