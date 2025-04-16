import torch
import torch.nn as nn


class TimesNetBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super(TimesNetBlock, self).__init__()
        self.conv1 = nn.Conv1d(input_size, output_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(output_size, output_size, kernel_size=3, padding=1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return x
    

class RewardNet(nn.Module):
    def __init__(self, input_size=10, cnn_layers=13, output_size=3):
        super(RewardNet, self).__init__()
        self.blocks = nn.ModuleList([TimesNetBlock(input_size, input_size) for _ in range(cnn_layers)])
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(input_size * 20, 128),  # 20 = sequence length
            nn.GELU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # B, input_size, seq_len
        for block in self.blocks:
            x = block(x)
        x = x.permute(0, 2, 1)  # B, seq_len, input_size
        x = self.flatten(x)
        return self.fc(x)