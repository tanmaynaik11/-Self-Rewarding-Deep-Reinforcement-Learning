import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

def preprocess_and_scale_data(df, scaled_columns, columns_to_drop=None,
                               date_col="Date", date_format="%d-%m-%Y",
                               split_date="2020-12-31",
                               train_output="training_data.csv",
                               test_output="test_data.csv",
                               cleaned_test_output="cleaned_test_data.csv",
                               scaler_output="minmax_scaler.save"):
    """
    Preprocesses financial time series data by converting dates, scaling features,
    splitting into train/test, and dropping specified columns from test set.

    Parameters:
    ----------
    df : pd.DataFrame
        The full DataFrame containing daily + weekly features and a 'Date' column.
    scaled_columns : list of str
        Columns to scale using MinMaxScaler.
    columns_to_drop : list of str, optional
        Columns to drop from the test set after scaling.
    date_col : str, optional
        Name of the column containing date strings (default is "Date").
    date_format : str, optional
        Format of the date strings (default is "%d-%m-%Y").
    split_date : str, optional
        Date to split train and test sets (default is "2020-12-31").
    train_output : str
        File path to save the training data CSV.
    test_output : str
        File path to save the full test data CSV (before dropping columns).
    cleaned_test_output : str
        File path to save the cleaned test data CSV (after dropping columns).
    scaler_output : str
        File path to save the trained MinMaxScaler.

    Returns:
    -------
    None
    """
    # Drop any rows with missing values
    df = df.dropna()

    # Ensure date column is in datetime format
    df[date_col] = pd.to_datetime(df[date_col], format=date_format)

    # Split into train and test
    train_data = df[df[date_col] <= split_date].copy()
    test_data = df[df[date_col] > split_date].copy()

    # Fit scaler on training data
    scaler = MinMaxScaler()
    train_data[scaled_columns] = scaler.fit_transform(train_data[scaled_columns])
    test_data[scaled_columns] = scaler.transform(test_data[scaled_columns])

    # Save scaled data
    train_data.to_csv(train_output, index=False)
    test_data.to_csv(test_output, index=False)

    # Drop specified columns from test and save cleaned test data
    if columns_to_drop:
        test_data_cleaned = test_data.drop(columns=columns_to_drop, errors='ignore')
        test_data_cleaned.to_csv(cleaned_test_output, index=False)

    # Save the scaler
    joblib.dump(scaler, scaler_output)

    print(f"Preprocessing complete. Data and scaler saved to disk.")


# Usage
# scaled_columns = [
#     'Open', 'High', 'Low', 'Close', 'Volume',
#     'Open_weekly', 'High_weekly', 'Low_weekly',
#     'Close_weekly', 'Volume_weekly'
# ]

# columns_to_drop = ['Sharpe_Reward', 'Return_Reward', 'MinMax_Reward', 'Daily_Return']

# preprocess_and_scale_data(df, scaled_columns, columns_to_drop)