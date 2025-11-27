"""
Streamlit-ready Tic-Tac-Toe with a Deep Q-Network (DQN) agent.

How to run:
1. Install dependencies: pip install streamlit torch torchvision numpy
2. Run: streamlit run tic_tac_toe_dqn_streamlit.py

Features:
- Simple TicTacToe environment
- PyTorch DQN (MLP) with replay buffer
- Train inside the Streamlit app with configurable hyperparameters
- Play against the trained agent (human vs agent)
- Save / load model

Notes:
This implementation is intentionally small and educational. For better performance, more
engineering is recommended (target network updates, prioritized replay, better exploration).
"""

import streamlit as st
import numpy as np
import random
import collections
import torch
import torch.nn as nn
import torch.optim as optim
import os
from typing import List, Tuple

# ---------- Environment ----------
class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        # board: 0 empty, 1 agent (X), -1 opponent (O)
        self.board = np.zeros(9, dtype=np.int8)
        self.current_player = 1  # agent starts by default; we will let human move optionally
        self.terminated = False
        self.winner = None
        return self._get_state()

    def _get_state(self):
        return self.board.copy()

    def legal_actions(self):
        return [i for i in range(9) if self.board[i] == 0]

    def step(self, action: int, player: int):
        if self.terminated:
            raise ValueError("Game already terminated")
        if self.board[action] != 0:
            # illegal move
            self.terminated = True
            self.winner = -player  # punish the mover by giving opponent the win
            return self._get_state(), -1.0, True, {"illegal": True}

        self.board[action] = player
        done, winner = self._check_game_end()
        reward = 0.0
        if done:
            self.terminated = True
            self.winner = winner
            if winner == 0:
                reward = 0.5  # draw reward
            elif winner == 1:
                reward = 1.0
            else:
                reward = -1.0
        return self._get_state(), reward, done, {}

    def _check_game_end(self) -> Tuple[bool, int]:
        B = self.board.reshape(3,3)
        lines = []
        lines.extend(list(B))
        lines.extend(list(B.T))
        lines.append(np.array([B[i,i] for i in range(3)]))
        lines.append(np.array([B[i,2-i] for i in range(3)]))
        for line in lines:
            s = np.sum(line)
            if s == 3:
                return True, 1
            if s == -3:
                return True, -1
        if np.all(self.board != 0):
            return True, 0
        return False, None

    def render(self) -> List[str]:
        symbols = {1: 'X', -1: 'O', 0: ' '}
        return [symbols[int(x)] for x in self.board]

# ---------- DQN components ----------

def state_to_tensor(state: np.ndarray) -> torch.FloatTensor:
    # state shape (9,), values -1,0,1. Convert to float tensor
    return torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # shape (1,9)

class DQN(nn.Module):
    def __init__(self, input_dim=9, output_dim=9, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state.copy(), action, reward, next_state.copy() if next_state is not None else None, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.vstack(states), np.array(actions), np.array(rewards, dtype=np.float32), (
            np.vstack(next_states) if next_states[0] is not None else None), np.array(dones, dtype=np.uint8)

    def __len__(self):
        return len(self.buffer)

# ---------- Training utilities ----------

def select_action_epsilon_greedy(net: DQN, state: np.ndarray, legal_actions: List[int], eps: float) -> int:
    if random.random() < eps:
        return random.choice(legal_actions)
    net.eval()
    with torch.no_grad():
        qvals = net(state_to_tensor(state)).squeeze(0).cpu().numpy()
    # mask illegal actions to very low
    masked = np.full_like(qvals, -1e9, dtype=float)
    masked[legal_actions] = qvals[legal_actions]
    return int(np.argmax(masked))

@torch.no_grad()
def evaluate_policy(net: DQN, episodes=100) -> float:
    env = TicTacToe()
    wins = 0
    for _ in range(episodes):
        state = env.reset()
        done = False
        current_player = 1
        while not done:
            if current_player == 1:
                action = select_action_epsilon_greedy(net, state, env.legal_actions(), eps=0.0)
            else:
                action = random.choice(env.legal_actions())
            state, reward, done, _ = env.step(action, current_player)
            current_player *= -1
        if env.winner == 1:
            wins += 1
    return wins / episodes

# ---------- Streamlit app ----------

st.set_page_config(page_title='Tic-Tac-Toe DQN', layout='wide')

st.title('Tic-Tac-Toe with Deep Q-Network (DQN)')

# Sidebar: hyperparameters & controls
with st.sidebar:
    st.header('Training controls')
    episodes_to_train = st.number_input('Episodes to train (per click)', min_value=1, max_value=20000, value=500, step=100)
    batch_size = st.number_input('Batch size', min_value=8, max_value=512, value=64, step=8)
    lr = st.number_input('Learning rate', value=1e-3, format="%.6f")
    gamma = st.number_input('Discount factor (gamma)', value=0.99)
    eps_start = st.number_input('Eps start', value=1.0)
    eps_end = st.number_input('Eps end', value=0.05)
    eps_decay = st.number_input('Eps decay (episodes)', min_value=1, value=1000)
    buffer_size = st.number_input('Replay buffer size', min_value=100, max_value=200000, value=10000, step=100)
    hidden = st.number_input('Network hidden units', min_value=16, max_value=512, value=128, step=16)
    save_path = st.text_input('Model save path', value='tic_tac_toe_dqn.pt')
    st.markdown('---')
    st.write('Agent vs Random Opponent by default. You can play against the agent from the main area.')

# Session state for model, buffer, optimizer, env, training stats
if 'net' not in st.session_state:
    st.session_state.net = DQN(input_dim=9, output_dim=9, hidden=int(hidden))
    st.session_state.target_net = DQN(input_dim=9, output_dim=9, hidden=int(hidden))
    st.session_state.target_net.load_state_dict(st.session_state.net.state_dict())
    st.session_state.optimizer = optim.Adam(st.session_state.net.parameters(), lr=lr)
    st.session_state.replay = ReplayBuffer(capacity=int(buffer_size))
    st.session_state.trained_episodes = 0
    st.session_state.loss = 0.0
    st.session_state.eps = eps_start
    st.session_state.hidden = int(hidden)

# If user changed hidden size, reinitialize networks
if st.session_state.hidden != int(hidden):
    st.session_state.net = DQN(input_dim=9, output_dim=9, hidden=int(hidden))
    st.session_state.target_net = DQN(input_dim=9, output_dim=9, hidden=int(hidden))
    st.session_state.target_net.load_state_dict(st.session_state.net.state_dict())
    st.session_state.optimizer = optim.Adam(st.session_state.net.parameters(), lr=lr)
    st.session_state.replay = ReplayBuffer(capacity=int(buffer_size))
    st.session_state.trained_episodes = 0
    st.session_state.hidden = int(hidden)

# Update optimizer lr if changed
for g in st.session_state.optimizer.param_groups:
    g['lr'] = lr

# Main area: left - training, right - play
col1, col2 = st.columns([2,1])

with col1:
    st.subheader('Training')
    train_button = st.button('Train')
    save_button = st.button('Save model')
    load_button = st.button('Load model')

    if save_button:
        torch.save(st.session_state.net.state_dict(), save_path)
        st.success(f'Saved model to {save_path}')

    if load_button:
        if os.path.exists(save_path):
            st.session_state.net.load_state_dict(torch.load(save_path))
            st.session_state.target_net.load_state_dict(st.session_state.net.state_dict())
            st.success(f'Loaded model from {save_path}')
        else:
            st.error('Model file not found')

    if train_button:
        pbar = st.progress(0)
        losses = []
        for ep in range(int(episodes_to_train)):
            env = TicTacToe()
            state = env.reset()
            done = False
            player = 1  # agent plays as 1
            # play an episode: agent plays first, opponent random
            while not done:
                legal = env.legal_actions()
                action = select_action_epsilon_greedy(st.session_state.net, state, legal, st.session_state.eps)
                next_state, reward, done, info = env.step(action, player)
                if 'illegal' in info:
                    # punish illegal move
                    st.session_state.replay.push(state, action, -1.0, next_state, True)
                    break
                if not done:
                    # opponent random move
                    opp_legal = env.legal_actions()
                    if len(opp_legal) > 0:
                        opp_action = random.choice(opp_legal)
                        next_state2, reward2, done2, _ = env.step(opp_action, -player)
                        if done2:
                            # if opponent wins, agent gets -1
                            final_reward = -1.0 if env.winner == -1 else (0.5 if env.winner == 0 else 0)
                            st.session_state.replay.push(next_state, opp_action, final_reward, next_state2, True)
                            # push transition for agent's previous move
                            st.session_state.replay.push(state, action, final_reward, next_state2, True)
                            state = next_state2
                            break
                        else:
                            # continue
                            st.session_state.replay.push(state, action, 0.0, next_state2, False)
                            state = next_state2
                    else:
                        # draw after agent move?
                        st.session_state.replay.push(state, action, 0.5, next_state, True)
                        state = next_state
                        break
                else:
                    # agent finished the game
                    st.session_state.replay.push(state, action, reward, next_state, True)
                    state = next_state
                    break

            # training step
            if len(st.session_state.replay) >= int(batch_size):
                states_b, actions_b, rewards_b, next_states_b, dones_b = st.session_state.replay.sample(int(batch_size))
                states_t = torch.tensor(states_b, dtype=torch.float32)
                actions_t = torch.tensor(actions_b, dtype=torch.int64).unsqueeze(1)
                rewards_t = torch.tensor(rewards_b, dtype=torch.float32).unsqueeze(1)
                dones_t = torch.tensor(dones_b, dtype=torch.float32).unsqueeze(1)
                next_states_t = torch.tensor(next_states_b, dtype=torch.float32)

                q_values = st.session_state.net(states_t).gather(1, actions_t)
                with torch.no_grad():
                    next_q = st.session_state.target_net(next_states_t).max(1)[0].unsqueeze(1)
                    target_q = rewards_t + (1 - dones_t) * gamma * next_q

                loss = nn.functional.mse_loss(q_values, target_q)
                st.session_state.optimizer.zero_grad()
                loss.backward()
                st.session_state.optimizer.step()
                losses.append(loss.item())

            # decay epsilon linearly
            st.session_state.trained_episodes += 1
            frac = min(1.0, st.session_state.trained_episodes / eps_decay)
            st.session_state.eps = eps_start + frac * (eps_end - eps_start)

            # periodically update target network
            if st.session_state.trained_episodes % 50 == 0:
                st.session_state.target_net.load_state_dict(st.session_state.net.state_dict())

            if (ep + 1) % max(1, int(episodes_to_train // 10)) == 0:
                pbar.progress(int((ep + 1) / episodes_to_train * 100))

        st.session_state.loss = float(np.mean(losses)) if losses else 0.0
        st.success(f'Trained {episodes_to_train} episodes — Avg loss: {st.session_state.loss:.4f} — Total episodes: {st.session_state.trained_episodes} — Eps: {st.session_state.eps:.3f}')

    st.markdown('---')
    st.write('Training stats')
    st.write(f"Episodes trained: {st.session_state.trained_episodes}")
    st.write(f"Last recorded loss: {st.session_state.loss:.6f}")

with col2:
    st.subheader('Play vs Agent')
    if 'game_env' not in st.session_state:
        st.session_state.game_env = TicTacToe()
        st.session_state.human_turn = True
        st.session_state.human_is_x = True  # human uses O or X? We'll treat human as -1 by default

    play_reset = st.button('Reset game')
    human_first = st.checkbox('Human goes first', value=True)
    human_is_x = st.checkbox('Human plays X (agent plays O)', value=True)

    if play_reset:
        st.session_state.game_env = TicTacToe()
        st.session_state.human_turn = human_first
        st.session_state.human_is_x = human_is_x

    env = st.session_state.game_env
    board_symbols = env.render()

    # Show board as 3x3 buttons
    board_cols = st.columns(3)
    for r in range(3):
        for c in range(3):
            idx = r * 3 + c
            text = board_symbols[idx]
            disabled = text != ' ' or env.terminated
            if board_cols[c].button(text or ' ', key=f'play_{idx}', disabled=disabled):
                if env.terminated:
                    st.warning('Game already finished — reset to play again')
                else:
                    # human move
                    human_player = 1 if human_is_x else -1
                    if env.board[idx] == 0:
                        _, _, done, _ = env.step(idx, human_player)
                        if done:
                            pass
                        else:
                            # agent move (greedy)
                            legal = env.legal_actions()
                            if len(legal) > 0:
                                agent_action = select_action_epsilon_greedy(st.session_state.net, env._get_state(), legal, eps=0.0)
                                env.step(agent_action, -human_player)
                    else:
                        st.warning('Illegal move')

    st.write('')
    if env.terminated:
        if env.winner == 1:
            st.success('X wins!')
        elif env.winner == -1:
            st.success('O wins!')
        else:
            st.info('Draw')

    st.markdown('---')
    st.subheader('Quick Eval')
    if st.button('Evaluate agent vs random (100 episodes)'):
        score = evaluate_policy(st.session_state.net, episodes=100)
        st.write(f'Agent win rate vs random: {score*100:.1f}%')

st.markdown('---')
st.write('Implementation notes: This DQN is intentionally simple (no target-network soft updates, no prioritized replay). It is a good starting point to understand DQN applied to a small board game. For better performance: add a stable target network update schedule, double DQN, prioritized replay, and more training episodes.')
