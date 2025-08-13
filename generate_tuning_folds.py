#!/usr/bin/env python3
import json
from pathlib import Path
import argparse

def load_folds_from_selected(json_path: Path, model_key: str):
    with json_path.open() as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        raise ValueError(f"Expected a non-empty list in {json_path}")

    ids = []
    for item in data:
        if not isinstance(item, dict):
            raise TypeError(f"List items must be dicts, got {type(item)}")
        if "global_fold_id" in item:
            ids.append(item["global_fold_id"])
        elif "fold_id" in item:
            ids.append(item["fold_id"])
        else:
            import re
            src = item.get("final_train_path") or item.get("train_path_arima") or ""
            m = re.search(r"_fold_(\d+)\.csv", src)
            if m:
                ids.append(int(m.group(1)))
            else:
                raise KeyError(f"Missing 'global_fold_id'/'fold_id' and cannot infer from paths in {item}")
    return ids


def main():
    parser = argparse.ArgumentParser(description="Generate tuning folds JSON from selected final paths")
    parser.add_argument("--model", required=True, choices=["arima", "lstm"], help="Model type")
    parser.add_argument("--base-dir", default="data/processed_folds/final",
                        help="Base directory where selected_<model>_final_paths.json is located")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    input_path = base_dir / args.model / f"selected_{args.model}_final_paths.json"
    output_path = base_dir / args.model / f"{args.model}_tuning_folds.json"

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    fold_ids = load_folds_from_selected(input_path, args.model)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(fold_ids, f, indent=2)

    print(f"[EMIT] Wrote {len(fold_ids)} tuning fold IDs to {output_path}")


if __name__ == "__main__":
    main()