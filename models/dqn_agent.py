import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class DQNAgent(nn.Module):
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.9):
        super(DQNAgent, self).__init__()
        self.qnet = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.memory = deque(maxlen=1000)
        self.gamma = gamma
        self.epsilon = 0.9  # for exploration

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 2)  # 3 actions: Sell, Hold, Buy
        with torch.no_grad():
            return self.qnet(state).argmax().item()

    def remember(self, s, a, r, s_):
        self.memory.append((s, a, r, s_))

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        for s, a, r, s_ in batch:
            target = r
            if s_ is not None:
                target += self.gamma * torch.max(self.qnet(s_)).item()
            pred = self.qnet(s)[a]
            loss = self.loss_fn(pred, torch.tensor(target, dtype=torch.float32))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()