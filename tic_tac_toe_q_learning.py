import numpy as np
import random
from collections import defaultdict
from functools import lru_cache

EMPTY = 0
X = 1  # RL agent
O = 2  # opponent

#-----------------------------
# Winner check
#-----------------------------
def check_winner(board):
    wins = [
        (0,1,2),(3,4,5),(6,7,8),
        (0,3,6),(1,4,7),(2,5,8),
        (0,4,8),(2,4,6)
    ]
    for a,b,c in wins:
        if board[a] != EMPTY and board[a] == board[b] == board[c]:
            return board[a]
    if EMPTY not in board:
        return 3  # draw
    return 0      # ongoing

#-----------------------------
# Symmetry transforms
#-----------------------------
TRANSFORMS = [
    lambda b: b,
    lambda b: [b[i] for i in [6,3,0,7,4,1,8,5,2]],  # rot90
    lambda b: [b[i] for i in [8,7,6,5,4,3,2,1,0]],  # rot180
    lambda b: [b[i] for i in [2,5,8,1,4,7,0,3,6]],  # rot270
    lambda b: [b[i] for i in [2,1,0,5,4,3,8,7,6]],  # reflect horizontal
    lambda b: [b[i] for i in [6,7,8,3,4,5,0,1,2]],  # reflect vertical
    lambda b: [b[i] for i in [0,3,6,1,4,7,2,5,8]],  # reflect diag
    lambda b: [b[i] for i in [8,5,2,7,4,1,6,3,0]]   # reflect anti-diag
]

@lru_cache(None)
def canonical(board_tuple):
    board = list(board_tuple)
    candidates = [tuple(t(board)) for t in TRANSFORMS]
    return min(candidates)

#-----------------------------
# Q-table + actions
#-----------------------------
Q = defaultdict(float)

def available_moves(board):
    return [i for i in range(9) if board[i] == EMPTY]

def select_action(board, epsilon):
    moves = available_moves(board)
    if random.random() < epsilon:
        return random.choice(moves)

    c = canonical(tuple(board))
    vals = [Q[(c, m)] for m in moves]
    return moves[int(np.argmax(vals))]

def update_q(prev_board, action, reward, next_board, alpha=0.3, gamma=0.99):
    c_prev = canonical(tuple(prev_board))
    c_next = canonical(tuple(next_board))

    old = Q[(c_prev, action)]
    future = 0

    if reward == 0 and available_moves(next_board):
        future = max(Q[(c_next, a)] for a in available_moves(next_board))

    Q[(c_prev, action)] = old + alpha * (reward + gamma * future - old)

#-----------------------------
# SELF-PLAY TRAINING
#-----------------------------
def train_selfplay(episodes=30000, epsilon_decay=0.9995, min_epsilon=0.05):
    epsilon = 1.0
    for ep in range(episodes):
        board = [EMPTY]*9
        turn = X
        history = []

        while True:
            action = select_action(board, epsilon)
            prev = board.copy()
            board[action] = turn
            history.append((prev, action, turn))

            w = check_winner(board)
            if w != 0:
                for b,a,p in history:
                    if w == 3: r = 0
                    elif w == p: r = 1
                    else: r = -1
                    update_q(b, a, r, board)
                break

            turn = O if turn == X else X

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    print("Training complete!")

#-----------------------------
# MINIMAX PERFECT-PLAY OPPONENT
#-----------------------------
def minimax(board, player):
    w = check_winner(board)
    if w == X: return 1
    if w == O: return -1
    if w == 3: return 0

    moves = available_moves(board)
    scores = []
    for m in moves:
        new_board = board.copy()
        new_board[m] = player
        score = minimax(new_board, O if player == X else X)
        scores.append(score)

    return max(scores) if player == X else min(scores)

def minimax_action(board):
    best_score = 999
    best_move = None
    for m in available_moves(board):
        new_board = board.copy()
        new_board[m] = O
        score = minimax(new_board, X)
        if score < best_score:
            best_score = score
            best_move = m
    return best_move

#-----------------------------
# EVALUATION
#-----------------------------
def evaluate_vs_minimax(games=200):
    wins=0; losses=0; draws=0

    for _ in range(games):
        board = [EMPTY]*9
        turn = X

        while True:
            if turn == X:
                action = select_action(board, epsilon=0.0)
            else:
                action = minimax_action(board)

            board[action] = turn

            w = check_winner(board)
            if w != 0:
                if w == X: wins += 1
                elif w == O: losses += 1
                else: draws += 1
                break

            turn = O if turn == X else X

    return wins, losses, draws

#-----------------------------
# RUN TRAINING
#-----------------------------
if __name__ == "__main__":
    train_selfplay(episodes=30000)

    w, l, d = evaluate_vs_minimax()
    print(f"Against minimax → Wins: {w}, Losses: {l}, Draws: {d}")
