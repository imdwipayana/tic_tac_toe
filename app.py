import streamlit as st
import random

st.set_page_config(page_title="Tic-Tac-Toe", layout="centered")

# ---- Style the buttons (bigger, nicer) ----
st.markdown("""
    <style>
        div.stButton > button {
            height: 80px;
            width: 80px;
            font-size: 35px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🎮 Tic-Tac-Toe")
st.write("Play against a simple AI!")


# ---- Initialize session state ----
if "board" not in st.session_state:
    st.session_state.board = [""] * 9
if "player_turn" not in st.session_state:
    st.session_state.player_turn = True
if "game_over" not in st.session_state:
    st.session_state.game_over = False
if "message" not in st.session_state:
    st.session_state.message = ""


# ---- Game Logic ----
def check_winner(b):
    win_patterns = [
        (0,1,2),(3,4,5),(6,7,8),
        (0,3,6),(1,4,7),(2,5,8),
        (0,4,8),(2,4,6)
    ]
    for i, j, k in win_patterns:
        if b[i] == b[j] == b[k] and b[i] != "":
            return b[i]
    if "" not in b:
        return "Draw"
    return None


def ai_move():
    empty = [i for i, v in enumerate(st.session_state.board) if v == ""]
    if empty:
        return random.choice(empty)
    return None


# ---- Draw a cell ----
def draw_cell(pos, col):
    symbol = st.session_state.board[pos] if st.session_state.board[pos] else " "
    clicked = col.button(symbol, key=f"cell_{pos}")

    if clicked and not st.session_state.game_over:
        if st.session_state.board[pos] == "" and st.session_state.player_turn:
            st.session_state.board[pos] = "X"
            st.session_state.player_turn = False
            st.experimental_rerun()   # Refresh UI right away


# ---- Render board ----
for r in range(3):
    cols = st.columns(3)
    for c in range(3):
        idx = r * 3 + c
        draw_cell(idx, cols[c])


# ---- Check winner after X moves ----
winner = check_winner(st.session_state.board)
if winner and not st.session_state.game_over:
    st.session_state.game_over = True
    st.session_state.message = (
        f"🎉 Winner: {winner}" if winner != "Draw" else "🤝 It's a draw!"
    )
else:
    # ---- AI move ----
    if not st.session_state.player_turn and not st.session_state.game_over:
        ai_idx = ai_move()
        if ai_idx is not None:
            st.session_state.board[ai_idx] = "O"

        st.session_state.player_turn = True
        st.experimental_rerun()   # Refresh after AI move


# ---- Status ----
st.subheader(st.session_state.message)


# ---- Reset ----
if st.button("🔄 Reset Game"):
    st.session_state.board = [""] * 9
    st.session_state.player_turn = True
    st.session_state.game_over = False
    st.session_state.message = ""
    st.experimental_rerun()
