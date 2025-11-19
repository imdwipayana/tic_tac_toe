import streamlit as st
import random

st.set_page_config(page_title="Tic Tac Toe", layout="centered")

# Initialize board state
if "board" not in st.session_state:
    st.session_state.board = [""] * 9
if "turn" not in st.session_state:
    st.session_state.turn = "X"
if "game_over" not in st.session_state:
    st.session_state.game_over = False
if "winner" not in st.session_state:
    st.session_state.winner = None

def check_winner(board):
    win_patterns = [
        (0,1,2), (3,4,5), (6,7,8),  # rows
        (0,3,6), (1,4,7), (2,5,8),  # cols
        (0,4,8), (2,4,6)            # diagonals
    ]
    for a,b,c in win_patterns:
        if board[a] == board[b] == board[c] and board[a] != "":
            return board[a]
    return None

def ai_move():
    empty_cells = [i for i,v in enumerate(st.session_state.board) if v == ""]
    if empty_cells:
        return random.choice(empty_cells)
    return None

def handle_click(i):
    if st.session_state.game_over:
        return

    # User move
    if st.session_state.board[i] == "":
        st.session_state.board[i] = "X"

        winner = check_winner(st.session_state.board)
        if winner:
            st.session_state.game_over = True
            st.session_state.winner = winner
            st.rerun()

        # AI move
        ai_idx = ai_move()
        if ai_idx is not None:
            st.session_state.board[ai_idx] = "O"

        winner = check_winner(st.session_state.board)
        if winner:
            st.session_state.game_over = True
            st.session_state.winner = winner

        st.rerun()

st.title("🎮 Tic Tac Toe (Streamlit Edition)")

cols = st.columns(3)

for i in range(9):
    c = cols[i % 3]
    with c:
        if st.button(st.session_state.board[i] or " ", key=f"btn{i}", height=80):
            handle_click(i)

st.write("---")

# Game status message
if st.session_state.game_over:
    if st.session_state.winner:
        st.success(f"🎉 Winner: {st.session_state.winner}")
    else:
        st.info("It's a draw!")
else:
    st.caption("Your move!")

# Reset button
if st.button("🔄 Restart Game"):
    st.session_state.board = [""] * 9
    st.session_state.game_over = False
    st.session_state.winner = None
    st.rerun()
