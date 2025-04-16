import torch
import pandas as pd

df_test = pd.read_csv('data/cleaned_test_data.csv')

def predict_rewards(rewardnet, df_test, seq_len=20):
    rewardnet.eval()
    features = df_test[['Open', 'High', 'Low', 'Close', 'Volume',
                        'Open_weekly', 'High_weekly', 'Low_weekly', 'Close_weekly', 'Volume_weekly']].values
    X = []
    dates = []

    for i in range(len(features) - seq_len):
        X.append(features[i:i+seq_len])
        dates.append(df_test.iloc[i+seq_len]["Date"])

    X_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        preds = rewardnet(X_tensor).numpy()

    df_rewards = pd.DataFrame(preds, columns=["r_sell", "r_hold", "r_buy"])
    df_rewards["Date"] = dates
    return df_rewards