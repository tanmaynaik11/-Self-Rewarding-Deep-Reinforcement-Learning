import yfinance as yf
import os

def download_index_data(indices, start_date="2007-01-01", end_date="2023-12-31", save_dir="data/daily_data"):
    """
    Downloads historical stock index data from Yahoo Finance and saves each index as a CSV file.

    Parameters:
    ----------
    indices : list of str
        List of Yahoo Finance index symbols (e.g., ["^HSI", "^FCHI"]).
    start_date : str, optional
        Start date for the historical data in 'YYYY-MM-DD' format (default is "2007-01-01").
    end_date : str, optional
        End date for the historical data in 'YYYY-MM-DD' format (default is "2023-12-31").
    save_dir : str, optional
        Directory to save the CSV files (default is "data").

    Returns:
    -------
    None
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for index in indices:
        print(f"Downloading {index}...")
        df = yf.download(index, start=start_date, end=end_date)
        csv_path = os.path.join(save_dir, f"{index.replace('^', '')}.csv")
        df.to_csv(csv_path)
        print(f"Saved to {csv_path}")


# Usage
# indices = ["^HSI", "^FCHI", "^KS11", "^IXIC", "^GSPC", "^DJI"]
# download_index_data(indices)