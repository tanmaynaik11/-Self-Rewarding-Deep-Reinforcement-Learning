
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class RewardNetDataset(Dataset):
    def __init__(self, df, seq_len=20):
        self.seq_len = seq_len
        self.features = df[['Open', 'High', 'Low', 'Close', 'Volume',
                            'Open_weekly', 'High_weekly', 'Low_weekly', 'Close_weekly', 'Volume_weekly']].values
        self.labels = df[['Sharpe_Reward', 'Return_Reward', 'MinMax_Reward']].values

        self.inputs, self.targets = [], []
        for i in range(len(self.features) - seq_len):
            self.inputs.append(self.features[i:i+seq_len])
            self.targets.append(self.labels[i+seq_len])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x, y