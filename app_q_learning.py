import streamlit as st
import numpy as np
import random

st.set_page_config(page_title="Tic Tac Toe RL", layout="centered")

# -------------------------------------------------------
# Utility functions
# -------------------------------------------------------

def symmetries(board):
    """Generate symmetries of the board."""
    b = np.array(board).reshape(3, 3)
    rots = [np.rot90(b, k) for k in range(4)]
    flips = [np.fliplr(r) for r in rots]
    syms = rots + flips
    return [tuple(s.flatten()) for s in syms]

def canonical(board):
    """Return the canonical representation (smallest lexicographically)."""
    return min(symmetries(board))

def board_to_state(board):
    return canonical(tuple(board))

def available_actions(board):
    return [i for i, v in enumerate(board) if v == 0]

def check_winner(board):
    b = np.array(board).reshape(3, 3)
    lines = np.concatenate([b, b.T, [b.diagonal()], [np.fliplr(b).diagonal()]])
    if any(np.all(line == 1) for line in lines):
        return 1
    if any(np.all(line == 2) for line in lines):
        return 2
    if 0 not in board:
        return 0
    return None

def minimax(board, player):
    winner = check_winner(board)
    if winner == 2:
        return 1
    if winner == 1:
        return -1
    if winner == 0:
        return 0

    best = -float("inf") if player == 2 else float("inf")

    for a in available_actions(board):
        board[a] = player
        val = minimax(board, 2 if player == 1 else 1)
        board[a] = 0

        if player == 2:  # agent maximizing
            best = max(best, val)
        else:  # human minimizing
            best = min(best, val)

    return best

# -------------------------------------------------------
# Q-learning
# -------------------------------------------------------

def choose_action(board, Q, epsilon):
    actions = available_actions(board)
    if random.random() < epsilon:
        return random.choice(actions)
    state = board_to_state(board)
    if state not in Q:
        Q[state] = np.zeros(9)
    qvals = Q[state]
    valid = [(a, qvals[a]) for a in actions]
    return max(valid, key=lambda x: x[1])[0]

def train_q_learning(episodes=30000, alpha=0.1, gamma=0.99, epsilon=0.2):
    Q = {}
    for _ in range(episodes):
        board = [0] * 9
        player = 1
        history = []

        while True:
            state = board_to_state(board)
            if state not in Q:
                Q[state] = np.zeros(9)

            if player == 2:
                action = choose_action(board, Q, epsilon)
            else:
                action = random.choice(available_actions(board))

            history.append((state, action))

            board[action] = player
            winner = check_winner(board)
            if winner is not None:
                if winner == 2:
                    reward = 1
                elif winner == 1:
                    reward = -1
                else:
                    reward = 0

                for s, a in reversed(history):
                    if s not in Q:
                        Q[s] = np.zeros(9)
                    Q[s][a] += alpha * (reward - Q[s][a])
                    reward *= gamma
                break

            player = 3 - player

    return Q

# -------------------------------------------------------
# Streamlit State Initialization
# -------------------------------------------------------

if "Q" not in st.session_state:
    with st.spinner("Training RL agent... (30k games)"):
        st.session_state.Q = train_q_learning()
    st.success("Agent trained!")

if "board" not in st.session_state:
    st.session_state.board = [0] * 9

if "turn" not in st.session_state:
    st.session_state.turn = "human"

if "gameover" not in st.session_state:
    st.session_state.gameover = False

Q = st.session_state.Q
board = st.session_state.board

# -------------------------------------------------------
# Agent Move
# -------------------------------------------------------
def agent_move():
    if st.session_state.gameover:
        return
    state = board_to_state(st.session_state.board)
    if state not in Q:
        Q[state] = np.zeros(9)
    action = choose_action(st.session_state.board, Q, epsilon=0.0)
    st.session_state.board[action] = 2

    winner = check_winner(st.session_state.board)
    if winner is not None:
        st.session_state.gameover = True
        return

    st.session_state.turn = "human"

# -------------------------------------------------------
# UI Layout
# -------------------------------------------------------
st.title("🎮 Tic Tac Toe — Reinforcement Learning Agent")

col1, col2, col3 = st.columns(3)
buttons = []

# render board
for i, col in enumerate([col1, col2, col3]):
    with col:
        for j in range(3):
            idx = i * 3 + j
            label = " " if board[idx] == 0 else ("X" if board[idx] == 1 else "O")

            if st.button(label, key=f"btn{idx}", use_container_width=True, disabled=board[idx] != 0 or st.session_state.gameover):
                if st.session_state.turn == "human" and board[idx] == 0:
                    st.session_state.board[idx] = 1
                    st.session_state.turn = "agent"

                    winner = check_winner(st.session_state.board)
                    if winner is not None:
                        st.session_state.gameover = True
                    else:
                        agent_move()
                st.rerun()

# show status
winner = check_winner(board)
if winner == 1:
    st.error("😎 You won! Nice job!")
elif winner == 2:
    st.warning("🤖 Agent wins! It's learning well!")
elif winner == 0:
    st.info("🤝 It's a draw.")

# reset button
if st.button("🔄 Reset Game"):
    st.session_state.board = [0] * 9
    st.session_state.turn = "human"
    st.session_state.gameover = False
    st.rerun()
