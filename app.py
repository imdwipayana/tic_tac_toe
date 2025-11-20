import streamlit as st
import random
import numpy as np
import json
import os

# -----------------------------------------------
# Streamlit Page Setup
# -----------------------------------------------
st.set_page_config(page_title="Tic Tac Toe RL", layout="centered")

st.markdown("""
<style>
/* Game Grid Buttons */
div.stButton > button {
    height: 100px;
    width: 100px;
    font-size: 32px;
    font-weight: 600;
    border-radius: 16px !important;
}
.game-row {
    display: flex;
    justify-content: center;
    gap: 10px;
}
#title {
    text-align: center;
    font-size: 40px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------
# RL PARAMETERS
# -----------------------------------------------
ALPHA = 0.5
GAMMA = 0.9
EPSILON = 0.2

# -----------------------------------------------
# SESSION STATE INIT
# -----------------------------------------------
if "board" not in st.session_state:
    st.session_state.board = [""] * 9

if "game_over" not in st.session_state:
    st.session_state.game_over = False

if "winner" not in st.session_state:
    st.session_state.winner = None

if "Q" not in st.session_state:
    st.session_state.Q = {}  # q-table

if "last_ai_state" not in st.session_state:
    st.session_state.last_ai_state = None

if "last_ai_action" not in st.session_state:
    st.session_state.last_ai_action = None


# -----------------------------------------------
# Helper Functions
# -----------------------------------------------
def check_winner(board):
    win_patterns = [
        (0,1,2), (3,4,5), (6,7,8),
        (0,3,6), (1,4,7), (2,5,8),
        (0,4,8), (2,4,6),
    ]
    for a,b,c in win_patterns:
        if board[a] == board[b] == board[c] and board[a] != "":
            return board[a]
    if "" not in board:
        return "Tie"
    return None


def get_state(board):
    return "".join(board)


def available_actions(board):
    return [i for i, v in enumerate(board) if v == ""]


def get_q(state):
    if state not in st.session_state.Q:
        st.session_state.Q[state] = {}
    return st.session_state.Q[state]


# -----------------------------------------------
# RL: Choose Action
# -----------------------------------------------
def choose_action(board, difficulty="RL"):
    actions = available_actions(board)
    state = get_state(board)

    # ----- Difficulty Modes -----
    if difficulty == "Easy":
        return random.choice(actions)

    if difficulty == "Medium":
        if random.random() < 0.3:
            return random.choice(actions)

    if difficulty == "Hard":
        # look for immediate win
        for a in actions:
            board[a] = "O"
            if check_winner(board) == "O":
                board[a] = ""
                return a
            board[a] = ""

        # block opponent
        for a in actions:
            board[a] = "X"
            if check_winner(board) == "X":
                board[a] = ""
                return a
            board[a] = ""

    # ----- RL MODE -----
    qvals = get_q(state)

    # ε-greedy exploration
    if random.random() < EPSILON:
        return random.choice(actions)

    # Exploit Q-table
    q_list = [qvals.get(a, 0) for a in actions]
    return actions[int(np.argmax(q_list))]


# -----------------------------------------------
# RL Update
# -----------------------------------------------
def update_q(prev_state, action, reward, next_state):
    if action is None:
        return

    qvals = get_q(prev_state)
    next_qvals = get_q(next_state)

    old_q = qvals.get(action, 0)
    max_future_q = max(next_qvals.values()) if next_qvals else 0

    new_q = old_q + ALPHA * (reward + GAMMA * max_future_q - old_q)
    qvals[action] = new_q


# -----------------------------------------------
# AI Move
# -----------------------------------------------
def ai_move(difficulty="RL"):
    board = st.session_state.board
    state_before = get_state(board)

    action = choose_action(board, difficulty=difficulty)

    board[action] = "O"

    st.session_state.last_ai_state = state_before
    st.session_state.last_ai_action = action

    return action


# -----------------------------------------------
# HANDLE PLAYER MOVE
# -----------------------------------------------
def handle_click(i, difficulty="RL"):
    if st.session_state.game_over:
        return

    if st.session_state.board[i] != "":
        return

    # Player move
    st.session_state.board[i] = "X"

    winner = check_winner(st.session_state.board)
    if winner:
        st.session_state.game_over = True
        st.session_state.winner = winner

        if winner == "X":
            # punish AI's last move
            if st.session_state.last_ai_state is not None:
                update_q(
                    prev_state=st.session_state.last_ai_state,
                    action=st.session_state.last_ai_action,
                    reward=-1,
                    next_state="terminal"
                )

        st.rerun()

    # AI Move
    ai_idx = ai_move(difficulty=difficulty)

    winner = check_winner(st.session_state.board)
    if winner:
        st.session_state.game_over = True
        st.session_state.winner = winner

        if winner == "O":
            # reward AI
            update_q(
                prev_state=st.session_state.last_ai_state,
                action=st.session_state.last_ai_action,
                reward=1,
                next_state="terminal"
            )

        st.rerun()

    # small shaping reward
    update_q(
        prev_state=st.session_state.last_ai_state,
        action=st.session_state.last_ai_action,
        reward=0,
        next_state=get_state(st.session_state.board),
    )

    st.rerun()


# -----------------------------------------------
# TRAINING MODE (SELF-PLAY)
# -----------------------------------------------
def self_play_games(n=2000):
    for _ in range(n):
        board = [""] * 9
        turn = "X"
        last_state = None
        last_action = None

        while True:
            state = "".join(board)
            actions = [i for i, v in enumerate(board) if v == ""]

            if turn == "X":
                action = random.choice(actions)
                board[action] = "X"
            else:
                action = choose_action(board)
                if last_state is not None:
                    update_q(last_state, last_action, 0, state)
                last_state = state
                last_action = action
                board[action] = "O"

            winner = check_winner(board)
            if winner:
                reward = 1 if winner == "O" else -1
                update_q(last_state, last_action, reward, "terminal")
                break

            turn = "O" if turn == "X" else "X"


# -----------------------------------------------
# UI
# -----------------------------------------------
st.markdown("<h1 id='title'>🎮 Tic Tac Toe — Reinforcement Learning Edition</h1>", unsafe_allow_html=True)

# Difficulty selection
difficulty = st.selectbox("AI Difficulty", ["Easy", "Medium", "Hard", "RL (Learning)"])
difficulty = difficulty.replace("RL (Learning)", "RL")

# Game grid
for row in range(3):
    cols = st.columns(3, gap="small")
    for col in range(3):
        idx = row * 3 + col
        with cols[col]:
            if st.button(st.session_state.board[idx] or " ", key=f"cell_{idx}"):
                handle_click(idx, difficulty=difficulty)

st.write("---")

# Winner display
if st.session_state.game_over:
    if st.session_state.winner == "Tie":
        st.info("It's a tie!")
    else:
        st.success(f"Winner: {st.session_state.winner}")
else:
    st.caption("Your move!")

# Controls
colA, colB = st.columns(2)

with colA:
    if st.button("🔄 Restart Game"):
        st.session_state.board = [""] * 9
        st.session_state.game_over = False
        st.session_state.winner = None
        st.session_state.last_ai_state = None
        st.session_state.last_ai_action = None
        st.rerun()

with colB:
    if st.button("⚡ Train AI (2000 games)"):
        self_play_games()
        st.success("AI Trained!")

# Q-Table Viewer
st.write("---")
with st.expander("📊 Q-Table Viewer"):
    st.json(st.session_state.Q)
