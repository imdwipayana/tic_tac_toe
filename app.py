import streamlit as st
import random

st.set_page_config(page_title="Tic-Tac-Toe", layout="centered")

st.title("🎮 Tic-Tac-Toe")
st.write("Play against a simple AI!")

# Initialize board
if "board" not in st.session_state:
    st.session_state.board = [""] * 9
if "player_turn" not in st.session_state:
    st.session_state.player_turn = True
if "game_over" not in st.session_state:
    st.session_state.game_over = False
if "message" not in st.session_state:
    st.session_state.message = ""


def check_winner(b):
    wins = [
        (0,1,2), (3,4,5), (6,7,8),      # rows
        (0,3,6), (1,4,7), (2,5,8),      # cols
        (0,4,8), (2,4,6)                # diagonals
    ]
    for i,j,k in wins:
        if b[i] == b[j] == b[k] and b[i] != "":
            return b[i]
    if "" not in b:
        return "Draw"
    return None


def ai_move():
    """Random AI"""
    empty = [i for i, v in enumerate(st.session_state.board) if v == ""]
    if empty:
        return random.choice(empty)
    return None


col1, col2, col3 = st.columns(3)

def draw_button(pos, col):
    if st.session_state.board[pos] == "":
        if col.button(" ", key=pos, height=80):
            if st.session_state.player_turn and not st.session_state.game_over:
                st.session_state.board[pos] = "X"
                st.session_state.player_turn = False
    else:
        col.button(st.session_state.board[pos], key=pos, height=80, disabled=True)

# Board layout
for r in range(3):
    cols = st.columns(3)
    for c in range(3):
        idx = r * 3 + c
        draw_button(idx, cols[c])

# Check after player's move
winner = check_winner(st.session_state.board)

if winner and not st.session_state.game_over:
    st.session_state.game_over = True
    st.session_state.message = f"🎉 Winner: {winner}" if winner != "Draw" else "🤝 It's a draw!"


# AI move
if not st.session_state.player_turn and not st.session_state.game_over:
    ai = ai_move()
    if ai is not None:
        st.session_state.board[ai] = "O"
    st.session_state.player_turn = True

    winner = check_winner(st.session_state.board)
    if winner:
        st.session_state.game_over = True
        st.session_state.message = f"🎉 Winner: {winner}" if winner != "Draw" else "🤝 It's a draw!"


st.subheader(st.session_state.message)

if st.button("🔄 Reset Game"):
    st.session_state.board = [""] * 9
    st.session_state.player_turn = True
    st.session_state.game_over = False
    st.session_state.message = ""
