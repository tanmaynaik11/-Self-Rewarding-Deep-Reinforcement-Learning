from datasets.reward_dataset import RewardNetDataset
from models.rewardnet import RewardNet
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

def train_rewardnet(df, epochs=20, lr=0.0001, batch_size=32):
    dataset = RewardNetDataset(df)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = RewardNet()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    loss_history = []
    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            optimizer.zero_grad()
            preds = model(x)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    return model, loss_history


# Usage
# model, history = train_rewardnet(df)

# save model
# torch.save(model.state_dict(), "rewardnet_dji.pth")