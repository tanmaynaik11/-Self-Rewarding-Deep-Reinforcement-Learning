import pandas as pd
import os

def combine_index_data(index_name, daily_folder="data/cleaned", weekly_folder="data/weekly", output_folder="data/combined_indexes"):
    """
    Combines daily data with pre-generated weekly data and saves it to a CSV file.
    
    Parameters:
    - index_name (str): The name of the index (e.g., "^DJI")
    - daily_folder (str): Folder containing daily CSV files.
    - weekly_folder (str): Folder containing pre-generated weekly CSV files.
    - output_folder (str): Folder where the combined index data will be saved.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    try:
        # Load daily data
        daily_path = os.path.join(daily_folder, f"{index_name}_cleaned.csv")
        daily_df = pd.read_csv(daily_path, parse_dates=["Date"])
        
        # Load weekly data
        weekly_path = os.path.join(weekly_folder, f"{index_name}_weekly.csv")
        weekly_df = pd.read_csv(weekly_path, parse_dates=["Date"])
        
        # Set 'Date' as index for both DataFrames
        daily_df.set_index('Date', inplace=True)
        weekly_df.set_index('Date', inplace=True)
        
        # Merge weekly data with daily data
        combined_df = daily_df.join(weekly_df, how="left", rsuffix="_weekly")

        # Find the dates where weekly data exists (typically Fridays or last day of the week)
        weekly_dates = combined_df[combined_df['Open_weekly'].notna()].index.tolist()

        # For each weekly date, move the weekly data to the next valid day
        for date in weekly_dates:
            next_date_index = combined_df.index.get_loc(date) + 1  # Find the next row index
            
            if next_date_index < len(combined_df):  # Ensure index is within bounds
                # Move the weekly data one row down
                combined_df.iloc[next_date_index, -5:] = combined_df.loc[date, ['Open_weekly', 'High_weekly', 'Low_weekly', 'Close_weekly', 'Volume_weekly']]
                
                # Clear the original weekly data row
                combined_df.loc[date, ['Open_weekly', 'High_weekly', 'Low_weekly', 'Close_weekly', 'Volume_weekly']] = None

        # Forward fill the weekly data to cover all daily rows within the same week
        combined_df.fillna(method='ffill', inplace=True)
        
        # Save the combined data
        output_path = os.path.join(output_folder, f"{index_name}_combined.csv")
        combined_df.to_csv(output_path)
        
        print(f"Successfully saved combined data for {index_name} at {output_path}")
        return combined_df

    except Exception as e:
        print(f"Error processing {index_name}: {e}")
        return None
    


# Usage
# # List of stock indices
# indices = ["^HSI", "^FCHI", "^KS11", "^IXIC", "^GSPC", "^DJI"]

# # Combine each index's data and save separately
# for index_name in indices:
#     combine_index_data(index_name)