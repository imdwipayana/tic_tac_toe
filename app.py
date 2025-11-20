import streamlit as st
import random

st.set_page_config(page_title="Tic Tac Toe", layout="centered")

# Initialize session state
if "board" not in st.session_state:
    st.session_state.board = [""] * 9
if "game_over" not in st.session_state:
    st.session_state.game_over = False
if "winner" not in st.session_state:
    st.session_state.winner = None

def check_winner(board):
    win_patterns = [
        (0,1,2), (3,4,5), (6,7,8),
        (0,3,6), (1,4,7), (2,5,8),
        (0,4,8), (2,4,6)
    ]
    for a,b,c in win_patterns:
        if board[a] == board[b] == board[c] and board[a] != "":
            return board[a]
    return None

import json
import os
import numpy as np

# RL Parameters
ALPHA = 0.5
GAMMA = 0.9
EPSILON = 0.2

# Q-table (key: board_state as string, value: dict of action->Q)
if "Q" not in st.session_state:
    st.session_state.Q = {}

def get_state(board):
    return "".join(board)  # e.g. "XOXO...X"

def available_actions(board):
    return [i for i, v in enumerate(board) if v == ""]

def get_q(state):
    if state not in st.session_state.Q:
        st.session_state.Q[state] = {}
    return st.session_state.Q[state]

def choose_action(board):
    state = get_state(board)
    actions = available_actions(board)
    qvals = get_q(state)

    # Exploration
    if random.random() < EPSILON:
        return random.choice(actions)

    # Exploitation
    q_list = [qvals.get(a, 0) for a in actions]
    return actions[int(np.argmax(q_list))]

def update_q(prev_state, action, reward, next_state):
    qvals = get_q(prev_state)
    next_qvals = get_q(next_state)

    old_q = qvals.get(action, 0)
    max_future_q = max(next_qvals.values()) if next_qvals else 0

    new_q = old_q + ALPHA * (reward + GAMMA * max_future_q - old_q)
    qvals[action] = new_q


def ai_move():
    board = st.session_state.board
    prev_state = get_state(board)

    action = choose_action(board)
    board[action] = "O"

    # Check if AI won immediately
    if check_winner(board) == "O":
        update_q(prev_state, action, reward=1, next_state=get_state(board))
        return action

    # If not terminal, small negative reward to encourage winning sooner
    update_q(prev_state, action, reward=0, next_state=get_state(board))

    return action


def handle_click(i):
    if winner == "X":
    # Punish last AI move
    update_q(prev_state=get_state(st.session_state.board), action=None, reward=-1, next_state="terminal")

    if st.session_state.game_over:
        return

    if st.session_state.board[i] == "":
        st.session_state.board[i] = "X"

        winner = check_winner(st.session_state.board)
        if winner:
            st.session_state.game_over = True
            st.session_state.winner = winner
            st.rerun()

        ai_idx = ai_move()
        if ai_idx is not None:
            st.session_state.board[ai_idx] = "O"

        winner = check_winner(st.session_state.board)
        if winner:
            st.session_state.game_over = True
            st.session_state.winner = winner

        st.rerun()

st.title("🎮 Tic Tac Toe (Streamlit)")

# Add CSS to enlarge buttons
st.markdown("""
<style>
button[kind="secondary"] {
    height: 80px !important;
    font-size: 30px !important;
}
</style>
""", unsafe_allow_html=True)

cols = st.columns(3)

for i in range(9):
    c = cols[i % 3]
    with c:
        if st.button(st.session_state.board[i] or " ", key=f"btn{i}"):
            handle_click(i)

st.write("---")

if st.session_state.game_over:
    st.success(f"Winner: {st.session_state.winner}")
else:
    st.caption("Your move!")

if st.button("🔄 Restart Game"):
    st.session_state.board = [""] * 9
    st.session_state.game_over = False
    st.session_state.winner = None
    st.rerun()
