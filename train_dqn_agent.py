import DQNAgent
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque 

def train_srddqn_agent(df_combined, seq_len=20, episodes=100):
    state_dim = 10 * seq_len
    agent = DQNAgent(state_dim=state_dim, action_dim=3)
    rewards_log = []

    for ep in range(episodes):
        total_reward = 0
        for i in range(len(df_combined) - seq_len - 1):
            window = df_combined.iloc[i:i+seq_len]
            next_window = df_combined.iloc[i+1:i+seq_len+1]

            # State & Next State
            s = torch.tensor(window[['Open', 'High', 'Low', 'Close', 'Volume',
                                     'Open_weekly', 'High_weekly', 'Low_weekly', 'Close_weekly', 'Volume_weekly']].values.flatten(),
                             dtype=torch.float32)
            s_ = torch.tensor(next_window[['Open', 'High', 'Low', 'Close', 'Volume',
                                           'Open_weekly', 'High_weekly', 'Low_weekly', 'Close_weekly', 'Volume_weekly']].values.flatten(),
                             dtype=torch.float32)

            action = agent.act(s)

            # Use predicted reward from RewardNet
            reward = window.iloc[-1][['r_sell', 'r_hold', 'r_buy']].values[action]
            total_reward += reward

            agent.remember(s, action, reward, s_)
            agent.replay()

        rewards_log.append(total_reward)
        print(f"Episode {ep+1}/{episodes}, Total Reward: {total_reward:.2f}")

    return agent, rewards_log