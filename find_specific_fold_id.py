import json
import os

target_fold_id = 60

results_dir = 'data/tuning_results'
moga_results_path = os.path.join(results_dir, 'final_moga_prophet_results.json')

try:
    with open(moga_results_path, 'r') as f:
        moga_results = json.load(f)
    
    found_index = -1
    for index, fold_data in enumerate(moga_results):
        if fold_data['fold_id'] == target_fold_id:
            found_index = index
            break
    if found_index != -1:
        print(f"The index for fold_id {target_fold_id} is: {found_index}")
        print(f"You should set 'fold_to_plot_index = {found_index}' in your visualization script.")
    else:
        print(f"Fold ID {target_fold_id} was not found in your representative folds list.")

except FileNotFoundError:
    print(f"ERROR: MOGA results file not found at {moga_results_path}.")