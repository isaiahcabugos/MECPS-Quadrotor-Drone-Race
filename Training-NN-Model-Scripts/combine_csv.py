import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import pickle
import glob  # <--- IMPORT GLOB

# --- 1. Define Paths and Columns ---

# Directory where your ROS node saves data
DATA_DIR = os.path.expanduser("~/mpc_data")

# Use glob to find all files matching the 'run_*.csv' pattern
file_names = glob.glob(os.path.join(DATA_DIR, "run_*.csv")) # <--- UPDATED

# Define feature (input) and label (output) columns
FEATURES = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'xt', 'yt', 'zt']
LABELS = ['vmx', 'vmy', 'vmz']

# Directory to save the processed data and scalers
OUTPUT_DIR = os.path.join(DATA_DIR, "processed_drone_data") # <--- Save to a sub-directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Looking for data in: {DATA_DIR}")
print(f"Saving processed data to: {OUTPUT_DIR}")

# --- 2. Load and Combine All CSV Files ---
if not file_names: # <--- Added a check in case no files are found
    print(f"Error: No 'run_*.csv' files found in {DATA_DIR}.")
    print("Please make sure your ROS node has run and saved data.")
else:
    df_list = []
    print(f"Found {len(file_names)} CSV files. Loading and combining...")
    for f in file_names:
        try:
            # 'f' is now the full path, which is correct
            df_list.append(pd.read_csv(f))
        except pd.errors.EmptyDataError:
            print(f"Warning: Skipping empty file: {f}")
        except Exception as e:
            print(f"Warning: Could not read {f}. Error: {e}")

    if not df_list:
        print("Error: No data was successfully loaded from the CSV files.")
    else:
        # Combine all dataframes into one
        combined_df = pd.concat(df_list, ignore_index=True)
        print(f"Successfully combined all files. Total data points: {len(combined_df)}")
        
        # --- 3. Check for Missing Columns ---
        all_cols = FEATURES + LABELS
        missing_cols = [col for col in all_cols if col not in combined_df.columns]
        
        if missing_cols:
            print(f"Error: The following required columns are missing from the data: {missing_cols}")
        else:
            # --- 4. Separate Features (X) and Labels (y) ---
            X = combined_df[FEATURES]
            y = combined_df[LABELS]
            print("Separated features (X) and labels (y).")

            # --- 5. Split into Training and Test Sets (80/20) ---
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=0.2,    # 20% for testing
                random_state=42   # For reproducible splits
            )
            print("Data split into 80% training and 20% test sets.")
            print(f"  Training set size: {len(X_train)} samples")
            print(f"  Test set size:     {len(X_test)} samples")

            # --- 6. Scale the Data ---
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()

            # Fit the scaler ONLY on the training data
            X_train_scaled = scaler_X.fit_transform(X_train)
            y_train_scaled = scaler_y.fit_transform(y_train)

            # Use the SAME (fitted) scaler to transform the test data
            X_test_scaled = scaler_X.transform(X_test)
            y_test_scaled = scaler_y.transform(y_test)
            
            print("Data scaling complete (fit on train, transformed train and test).")

            # --- 7. Save the Scalers ---
            scaler_x_path = os.path.join(OUTPUT_DIR, 'scaler_X.pkl')
            scaler_y_path = os.path.join(OUTPUT_DIR, 'scaler_y.pkl')
            
            with open(scaler_x_path, 'wb') as f:
                pickle.dump(scaler_X, f)
            with open(scaler_y_path, 'wb') as f:
                pickle.dump(scaler_y, f)
            print(f"Saved X scaler to: {scaler_x_path}")
            print(f"Saved y scaler to: {scaler_y_path}")

            # --- 8. Save the Processed Datasets ---
            
            # Re-combine scaled features and labels into DataFrames
            train_scaled_df = pd.DataFrame(X_train_scaled, columns=FEATURES)
            train_scaled_df[LABELS] = y_train_scaled

            test_scaled_df = pd.DataFrame(X_test_scaled, columns=FEATURES)
            test_scaled_df[LABELS] = y_test_scaled
            
            # Define output paths
            train_csv_path = os.path.join(OUTPUT_DIR, 'train_scaled_dataset.csv')
            test_csv_path = os.path.join(OUTPUT_DIR, 'test_scaled_dataset.csv')

            # Save to CSV
            train_scaled_df.to_csv(train_csv_path, index=False)
            test_scaled_df.to_csv(test_csv_path, index=False)

            print(f"Saved scaled training data to: {train_csv_path}")
            print(f"Saved scaled test data to: {test_csv_path}")
            print("\nData preparation complete!")