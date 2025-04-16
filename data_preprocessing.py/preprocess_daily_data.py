import pandas as pd
import os

def preprocess_daily_data(indices, input_folder="data/daily_data", output_folder="data/cleaned"):
    """
    Preprocesses daily data for multiple stock indices by:
    - Detecting incorrect headers and renaming columns
    - Fixing date format (DD-MM-YYYY → datetime)
    - Ensuring numeric data types
    - Rearranging columns to match weekly format
    - Saving cleaned data
    """
    os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists

    # Correct column order (excluding "Date" because it becomes the index)
    correct_order = ["Open", "High", "Low", "Close", "Volume"]

    for index in indices:
        file_path = f"{input_folder}/{index}.csv"
        output_path = f"{output_folder}/{index}_cleaned.csv"

        try:
            # Load data and print available columns for debugging
            df = pd.read_csv(file_path)
            print(f"Processing {index}... Columns found: {df.columns.tolist()}")

            # Ensure "Date" is the first column
            if "Date" not in df.columns:
                df.rename(columns={df.columns[0]: "Date"}, inplace=True)

            # Convert Date column to datetime format
            df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", errors="coerce")

            # Drop rows where Date is NaT (if conversion failed)
            df.dropna(subset=["Date"], inplace=True)

            # Set Date as index
            df.set_index("Date", inplace=True)

            # Ensure correct column data types
            df = df.astype({"Open": float, "High": float, "Low": float, "Close": float, "Volume": int}, errors="ignore")

            # Reorder columns to match weekly format (excluding "Date" since it's the index)
            df = df[correct_order]

            # Save cleaned data
            df.to_csv(output_path)
            print(f"Cleaned data saved for {index} at {output_path}")

        except Exception as e:
            print(f"Error processing {index}: {e}")


# Usage
# # List of stock indices to process
# indices = ["^HSI", "^FCHI", "^KS11", "^IXIC", "^GSPC", "^DJI"]

# # Run preprocessing
# preprocess_daily_data(indices)