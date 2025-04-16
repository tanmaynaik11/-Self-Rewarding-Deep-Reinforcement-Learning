import pandas as pd

def merge_rewards_with_data(df_test, df_rewards):
    df_test = df_test.iloc[20:].reset_index(drop=True)  # Align with reward predictions
    df_combined = pd.concat([df_test, df_rewards.drop(columns=["Date"])], axis=1)
    return df_combined