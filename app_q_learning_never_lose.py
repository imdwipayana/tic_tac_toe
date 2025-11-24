import streamlit as st
import numpy as np
import random
import pickle
import os

st.set_page_config(page_title="Tic Tac Toe RL", layout="centered")

# -------------------------------------------------------
# Tic Tac Toe Helper Functions
# -------------------------------------------------------

def symmetries(board):
    b = np.array(board).reshape(3, 3)
    rots = [np.rot90(b, k) for k in range(4)]
    flips = [np.fliplr(r) for r in rots]
    syms = rots + flips
    return [tuple(s.flatten()) for s in syms]

def canonical(board):
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

# -------------------------------------------------------
# Opponent Policies
# -------------------------------------------------------

def opponent_random(board):
    return random.choice(available_actions(board))

def opponent_selfplay(board, Q, epsilon):
    """Opponent: ε-greedy self-play."""
    acts = available_actions(board)
    state = board_to_state(board)
    if random.random() < epsilon:
        return random.choice(acts)
    if state in Q:
        qvals = Q[state]
        valid = [(a, qvals[a]) for a in acts]
        return max(valid, key=lambda x: x[1])[0]
    return random.choice(acts)

# -------------------------------------------------------
# Q-Learning
# -------------------------------------------------------

def choose_action(board, Q, epsilon):
    acts = available_actions(board)
    state = board_to_state(board)
    if random.random() < epsilon:
        return random.choice(acts)
    if state not in Q:
        Q[state] = np.zeros(9)
    qvals = Q[state]
    valid = [(a, qvals[a]) for a in acts]
    return max(valid, key=lambda x: x[1])[0]

def train_phase(Q, episodes, opponent, epsilon_agent, epsilon_opp, alpha=0.1, gamma=0.99):
    """Runs one phase of training."""
    for _ in range(episodes):
        board = [0] * 9
        player = 1
        history = []

        while True:
            if player == 2:  
                action = choose_action(board, Q, epsilon_agent)
            else:
                action = opponent(board, Q) if opponent != opponent_random else opponent(board)

            state = board_to_state(board)
            history.append((state, action))

            board[action] = player
            winner = check_winner(board)

            if winner is not None:
                reward = 1 if winner == 2 else (-1 if winner == 1 else 0)
                for s, a in reversed(history):
                    if s not in Q:
                        Q[s] = np.zeros(9)
                    Q[s][a] += alpha * (reward - Q[s][a])
                    reward *= gamma
                break

            player = 3 - player

    return Q

def train_full_curriculum():
    Q = {}

    st.write("### Phase 1: Training vs random (60k)")
    Q = train_phase(Q, episodes=60000, opponent=opponent_random,
                    epsilon_agent=0.4, epsilon_opp=1.0)

    st.write("### Phase 2: Self-play ε=0.3 (150k)")
    Q = train_phase(Q, episodes=150000,
                    opponent=lambda b, q: opponent_selfplay(b, q, 0.3),
                    epsilon_agent=0.25, epsilon_opp=0.3)

    st.write("### Phase 3: Near-greedy self-play ε=0.05 (100k)")
    Q = train_phase(Q, episodes=100000,
                    opponent=lambda b, q: opponent_selfplay(b, q, 0.05),
                    epsilon_agent=0.05, epsilon_opp=0.05)

    return Q


# -------------------------------------------------------
# Load / Train Q-table
# -------------------------------------------------------

Q_PATH = "q_table.pkl"

if os.path.exists(Q_PATH):
    with open(Q_PATH, "rb") as f:
        Q = pickle.load(f)
    st.success("Loaded pre-trained Q-table!")
else:
    st.warning("No Q-table found. Training from scratch…")
    with st.spinner("Training agent with curriculum learning… ~310k games"):
        Q = train_full_curriculum()
    with open(Q_PATH, "wb") as f:
        pickle.dump(Q, f)
    st.success("Training complete! Q-table saved.")

st.divider()

# -------------------------------------------------------
# Game State
# -------------------------------------------------------

if "board" not in st.session_state:
    st.session_state.board = [0] * 9
if "gameover" not in st.session_state:
    st.session_state.gameover = False
if "turn" not in st.session_state:
    st.session_state.turn = "human"

# -------------------------------------------------------
# Agent Move
# -------------------------------------------------------

def agent_move():
    if st.session_state.gameover:
        return
    state = board_to_state(st.session_state.board)
    if state not in Q:
        Q[state] = np.zeros(9)
    idx = choose_action(st.session_state.board, Q, 0.0)
    st.session_state.board[idx] = 2

    winner = check_winner(st.session_state.board)
    if winner is not None:
        st.session_state.gameover = True
        return

    st.session_state.turn = "human"


# -------------------------------------------------------
# UI Layout
# -------------------------------------------------------

st.title("🤖 Tic Tac Toe — Advanced RL Agent (Never Loses)")
board = st.session_state.board

cols = st.columns(3)
for i in range(3):
    with cols[i]:
        for j in range(3):
            idx = i * 3 + j
            label = " " if board[idx] == 0 else ("X" if board[idx] == 1 else "O")

            if st.button(label, key=f"b{idx}",
                         use_container_width=True, disabled=board[idx] != 0 or st.session_state.gameover):
                if st.session_state.turn == "human":
                    st.session_state.board[idx] = 1
                    st.session_state.turn = "agent"

                    winner = check_winner(st.session_state.board)
                    if winner is not None:
                        st.session_state.gameover = True
                    else:
                        agent_move()
                st.rerun()

# -------------------------------------------------------
# Status
# -------------------------------------------------------

winner = check_winner(board)
if winner == 1:
    st.error("😎 You beat the agent! (This should be really rare!)")
elif winner == 2:
    st.warning("🤖 Agent wins — flawless play!")
elif winner == 0:
    st.info("🤝 Draw.")

# -------------------------------------------------------
# Reset Button
# -------------------------------------------------------

if st.button("🔄 Reset Game"):
    st.session_state.board = [0] * 9
    st.session_state.turn = "human"
    st.session_state.gameover = False
    st.rerun()
