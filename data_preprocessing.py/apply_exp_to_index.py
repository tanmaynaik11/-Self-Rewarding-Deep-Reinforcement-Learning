import pandas as pd
import os
import calculate_expert_features 
def apply_expert_features_to_index(index_name, combined_folder="data/combined_indexes", output_folder="data/expert_features"):
    """
    Applies expert feature calculation to a specific index's combined dataset and saves it.
    
    Parameters:
    - index_name (str): Name of the index (e.g., "^DJI").
    - combined_folder (str): Folder where combined datasets are stored.
    - output_folder (str): Folder to save the processed dataset with expert features.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    try:
        # Load combined dataset
        file_path = os.path.join(combined_folder, f"{index_name}_combined.csv")
        data = pd.read_csv(file_path, parse_dates=["Date"])
        
        # Ensure the data is sorted by full datetime index (not just by day)
        data.sort_values(by="Date", inplace=True)
        
        # Apply expert feature calculation
        processed_data = calculate_expert_features(data)
        
        # Convert expert features to float type to ensure consistency
        processed_data['Sharpe_Reward'] = processed_data['Sharpe_Reward'].astype(float)
        processed_data['Return_Reward'] = processed_data['Return_Reward'].astype(float)
        processed_data['MinMax_Reward'] = processed_data['MinMax_Reward'].astype(float)
        
        # Save processed data with expert features
        output_path = os.path.join(output_folder, f"{index_name}_expert_features.csv")
        processed_data.to_csv(output_path, index=False)
        
        print(f"Expert features calculated and saved for {index_name} at {output_path}")
        return processed_data

    except Exception as e:
        print(f"Error processing {index_name}: {e}")
        return None
    

# Usage
# indices = ["^HSI", "^FCHI", "^KS11", "^IXIC", "^GSPC", "^DJI"]

# for index_name in indices:
#     apply_expert_features_to_index(index_name)