import os
import json
import argparse
from typing import Dict, Any, List

def prepare_tuning_folds(
    selected_json_path: str,
    folds_summary_path: str,
    output_path: str
) -> None:
    """
    Transforms the simplified JSON output of select_main.py into a more detailed
    format suitable for model tuning.

    Args:
        selected_json_path (str): Path to the JSON output file from select_main.py.
        folds_summary_path (str): Path to the original folds_summary_{model_type}.json file.
        output_path (str): Path to save the resulting JSON file.
    """
    try:
        with open(selected_json_path, "r") as f:
            selected_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Selected JSON file not found at: {selected_json_path}")
        return

    try:
        with open(folds_summary_path, "r") as f:
            folds_summary = json.load(f)
    except FileNotFoundError:
        print(f"Error: Original folds_summary file not found at: {folds_summary_path}")
        return

    # Create a dictionary for quick lookup of detailed fold information
    summary_map = {item["global_fold_id"]: item for item in folds_summary}
    
    tuning_folds = []
    found_count = 0
    
    for item in selected_data.get("selected_folds", []):
        gid = item.get("global_fold_id")
        if gid is not None and gid in summary_map:
            # Retrieve the full fold details from the summary map
            full_fold_info = summary_map[gid]
            tuning_folds.append(full_fold_info)
            found_count += 1
        else:
            print(f"Warning: global_fold_id {gid} not found in the original folds_summary file.")

    # Save the result
    with open(output_path, "w") as f:
        json.dump(tuning_folds, f, indent=2)

    print(f"Successfully created {os.path.basename(output_path)} with {found_count} selected folds.")

def main():
    parser = argparse.ArgumentParser(description="Transform the output of select_main.py into a detailed tuning folds format.")
    parser.add_argument("--model_type", required=True, choices=["lstm", "arima"], help="The model type (lstm or arima).")
    parser.add_argument("--selected_json_path", required=False, help="Path to the JSON output file from select_main.py (e.g., selected_lstm.json).",)
    parser.add_argument("--folds_summary_path", required=False, help="Path to the original folds_summary JSON file (e.g., folds_summary_lstm.json).",)
    parser.add_argument("--output_path", required=False, help="Path to save the resulting JSON file.")
    
    args = parser.parse_args()

    # Set default paths if not provided
    if not args.selected_json_path:
        args.selected_json_path = f"selected_{args.model_type}.json"
    if not args.folds_summary_path:
        args.folds_summary_path = f"folds_summary_{args.model_type}.json"
    if not args.output_path:
        args.output_path = f"tuning_folds_{args.model_type}.json"
        
    prepare_tuning_folds(
        args.selected_json_path,
        args.folds_summary_path,
        args.output_path
    )

if __name__ == "__main__":
    main()