import pandas as pd
import os

def add_index_label(index_name, folder_path, output_folder):
    """
    Adds an 'Index' column to the CSV file indicating which index the data represents.
    """
    os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists

    # Process both daily and weekly files
    for data_type in ['cleaned', 'weekly']:
        file_path = os.path.join(folder_path, data_type, f"{index_name}_{data_type}.csv")
        
        try:
            # Load CSV file
            df = pd.read_csv(file_path)
            
            # Add 'Index' column
            df['Index'] = index_name  # Adding a new column with the index name
            
            # Save the updated file
            output_file = os.path.join(output_folder, f"{index_name}_{data_type}_labeled.csv")
            df.to_csv(output_file, index=False)
            print(f"Labeled file saved for {index_name} at {output_file}")
            
        except Exception as e:
            print(f"Error processing {index_name}: {e}")


# usage
# # List of stock indices
# indices = ["^HSI", "^FCHI", "^KS11", "^IXIC", "^GSPC", "^DJI"]

# # Process each index to add labels
# for index_name in indices:
#     add_index_label(index_name, folder_path="data", output_folder="data/labeled")