import os
import json
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
# --- Helper Function: ADF Test ---
def perform_adf_test(series, fold_id):
    """
    Performs the Augmented Dickey-Fuller test on a series and returns the p-value.
    """
    # The adfuller function can't handle NaNs, so we ensure they are dropped.
    # Your 'Log_Returns' column should already have NaNs removed from its creation.
    result = adfuller(series.dropna())
    p_value = result[1]
    
    print(f"--- Fold ID: {fold_id} ---")
    print(f"ADF Test p-value: {p_value:.6f}")
    
    if p_value <= 0.05:
        print("Conclusion: The series is stationary.")
    else:
        print("Conclusion: The series is NOT stationary.")
    print("-" * 25)
    
    return p_value

# --- Step 1: Load your list of 75 representative fold IDs ---
# This is where you would load your 'representative_fold_ids.json' file.
# For this example, I will create a sample list of IDs.
# In your actual code, use this:
with open('data/processed_folds/shared_meta/representative_fold_ids.json', 'r') as f:
    representative_fold_ids = json.load(f)


adf_results = {}

for fold_id in representative_fold_ids:
    np.random.seed(fold_id) # Use fold_id for reproducibility
    log_returns_sample = pd.Series(np.random.randn(250) * 0.02) # Typical daily log returns
    fold_data = pd.DataFrame({'Log_Returns': log_returns_sample})

     # Isolate the 'Log_Returns' series
    log_returns_series = fold_data['Log_Returns']
    
    # Perform the ADF test and store the result
    p_value = perform_adf_test(log_returns_series, fold_id)
    adf_results[fold_id] = p_value
    
    # --- THIS IS THE PART YOU NEED TO ADAPT ---
    # Load the data for the specific fold_id.
    # This data should already contain your 'Log_Returns' column.
    # Example: fold_data = pd.read_csv(f'path/to/data/fold_{fold_id}.csv')
    
    # For demonstration, I'll create sample data for the fold
    # that mimics having a 'Log_Returns' column.
    np.random.seed(fold_id) # Use fold_id for reproducibility
    log_returns_sample = pd.Series(np.random.randn(250) * 0.02) # Typical daily log returns
    fold_data = pd.DataFrame({'Log_Returns': log_returns_sample})

    log_returns_series = fold_data['Log_Returns']
    
    # Perform the ADF test and store the result
    p_value = perform_adf_test(log_returns_series, fold_id)
    adf_results[fold_id] = p_value

print("\n--- Overall Summary ---")
stationary_folds = 0
non_stationary_folds = 0

for fold_id, p_value in adf_results.items():
    if p_value <= 0.05:
        stationary_folds += 1
    else:
        non_stationary_folds += 1

print(f"Total folds tested: {len(representative_fold_ids)}")
print(f"Number of stationary folds (p <= 0.05): {stationary_folds}")
print(f"Number of non-stationary folds (p > 0.05): {non_stationary_folds}")

if non_stationary_folds == 0:
    print("\nExcellent! All representative folds are stationary.")
    print("You can confidently proceed with using the 'Log_Returns' column for ARIMA, LSTM, and GRU.")
else:
    print("\nWarning: Some folds were found to be non-stationary.")
    print("You may need to investigate these specific folds further.")
