import pandas as pd
import os

def convert_daily_to_weekly(indices, daily_dir="data/cleaned", weekly_dir="data/weekly"):
    """
    Converts daily stock index data to weekly frequency and saves the results.

    Parameters:
    ----------
    indices : list of str
        List of Yahoo Finance index symbols (e.g., ["^HSI", "^FCHI"]).
    daily_dir : str, optional
        Directory containing cleaned daily CSV files (default is "data/cleaned").
    weekly_dir : str, optional
        Directory where the weekly resampled CSVs will be saved (default is "data/weekly").

    Returns:
    -------
    None
    """
    os.makedirs(weekly_dir, exist_ok=True)

    for index in indices:
        file_path = os.path.join(daily_dir, f"{index}_cleaned.csv")

        try:
            df_daily = pd.read_csv(file_path, parse_dates=['Date'])
            df_daily.set_index("Date", inplace=True)

            if "Close" not in df_daily.columns and "Adj Close" in df_daily.columns:
                df_daily.rename(columns={"Adj Close": "Close"}, inplace=True)

            df_weekly = df_daily.resample('W-FRI').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()

            weekly_file_path = os.path.join(weekly_dir, f"{index}_weekly.csv")
            df_weekly.to_csv(weekly_file_path)
            print(f"Saved weekly data for {index} at {weekly_file_path}")

        except Exception as e:
            print(f"Error processing {index}: {e}")


# Usage
# indices = ["^HSI", "^FCHI", "^KS11", "^IXIC", "^GSPC", "^DJI"]
# convert_daily_to_weekly(indices)