#!/usr/bin/env python3
# final_scaler_and_pca.py
import os
import json
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from dataclasses import dataclass
from joblib import dump
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

# -----------------------------
# IO ROOTS
# -----------------------------
BASE_DIR = "data"
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, "processed_folds")

FINAL_OUT_ROOT = os.path.join(OUTPUT_BASE_DIR, "final")
FINAL_MODELS_DIR = os.path.join(FINAL_OUT_ROOT, "models")
os.makedirs(FINAL_OUT_ROOT, exist_ok=True)
os.makedirs(FINAL_MODELS_DIR, exist_ok=True)

# -----------------------------
# Helpers / constants
# -----------------------------
NON_FEATURE_KEEP = ["Date", "Ticker", "target_log_returns", "target", "Log_Returns", "Close_raw"]

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def absolute_path_from_rel(rel_path: str) -> str:
    return os.path.join(OUTPUT_BASE_DIR, rel_path)

def read_df(rel_path: Optional[str]) -> Optional[pd.DataFrame]:
    if not rel_path:
        return None
    abspath = absolute_path_from_rel(rel_path)
    if not os.path.exists(abspath):
        return None
    return pd.read_csv(abspath)

def resolve_selected_list(selected_payload) -> List[Dict]:
    if isinstance(selected_payload, list):
        return selected_payload
    if isinstance(selected_payload, dict) and "selected_folds" in selected_payload:
        return selected_payload["selected_folds"]
    raise ValueError("[ERROR] selected_folds_json must be either a list or a dict with key 'selected_folds'.")

def build_gid_map(folds_summary_rows: List[Dict]) -> Dict[int, Dict]:
    m = {}
    for r in folds_summary_rows:
        gid = r.get("global_fold_id")
        if gid is not None:
            m[int(gid)] = r
    return m

def enrich_selected_with_paths(selected_list: List[Dict],
                               folds_summary_rows: List[Dict],
                               model_type: str) -> List[Dict]:
    gid_map = build_gid_map(folds_summary_rows)
    key_train = "train_path_lstm" if model_type == "lstm" else "train_path_arima"
    key_val   = "val_path_lstm"   if model_type == "lstm" else "val_path_arima"
    key_test  = "test_path_lstm"  if model_type == "lstm" else "test_path_arima"

    out = []
    missing = 0
    for item in selected_list:
        gid = item.get("global_fold_id")
        if gid is None:
            continue
        src = gid_map.get(int(gid))
        if not src:
            missing += 1
            continue
        merged = dict(src)
        merged.setdefault("ticker", item.get("ticker"))
        merged.setdefault("date_min", item.get("date_min"))
        merged.setdefault("date_max", item.get("date_max"))
        if not (merged.get(key_train) and merged.get(key_val)):
            missing += 1
            continue
        out.append(merged)
    if missing:
        print(f"[WARN] {missing} selected items could not be matched/enriched from folds_summary.")
    if not out:
        raise RuntimeError("[ERROR] No selected folds could be enriched with paths. Check --folds-summary-path.")
    return out

def infer_feature_columns_from_first_fold(selected_folds: List[Dict], model_type: str) -> List[str]:
    key_train = "train_path_lstm" if model_type == "lstm" else "train_path_arima"
    for fd in selected_folds:
        rel = fd.get(key_train)
        df = read_df(rel)
        if df is None:
            continue
        if model_type.lower() == "arima":
            if "Log_Returns" in df.columns and pd.api.types.is_numeric_dtype(df["Log_Returns"]):
                return ["Log_Returns"]
            if "Close_raw" in df.columns and pd.api.types.is_numeric_dtype(df["Close_raw"]):
                return ["Close_raw"]
            cols = [c for c in df.columns
                    if c not in ["Date", "Ticker"] and not str(c).lower().startswith("target")]
            numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
            if numeric_cols:
                return [numeric_cols[0]]
            continue
        # LSTM
        cols = [c for c in df.columns if c not in NON_FEATURE_KEEP and not str(c).lower().startswith("target")]
        cols = [c for c in cols if c not in ["Date", "Ticker", "Log_Returns", "Close_raw"]]
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            return numeric_cols
    raise RuntimeError("[ERROR] Unable to infer feature columns from selected folds.")

def gather_union_train(selected_folds: List[Dict],
                       model_type: str,
                       feature_cols: List[str]) -> pd.DataFrame:
    frames = []
    missing = skipped = 0
    key_train = "train_path_lstm" if model_type == "lstm" else "train_path_arima"
    for fd in selected_folds:
        tr_rel = fd.get(key_train)
        if not tr_rel:
            skipped += 1
            continue
        df = read_df(tr_rel)
        if df is None:
            missing += 1
            continue
        if any(c not in df.columns for c in feature_cols):
            skipped += 1
            continue
        frames.append(df[feature_cols])
    if missing:
        print(f"[WARN] Missing {missing} train files in selected folds.")
    if skipped:
        print(f"[WARN] Skipped {skipped} train files due to missing path or feature mismatch.")
    if not frames:
        raise RuntimeError("[ERROR] No valid TRAIN data found to fit scaler/PCA.")
    union_train = pd.concat(frames, axis=0, ignore_index=True)
    if not np.isfinite(union_train.to_numpy(dtype=np.float32)).all():
        raise ValueError("[ERROR] Non-finite values found in UNION TRAIN features.")
    return union_train

def drop_duplicate_log_returns(out_df: pd.DataFrame) -> pd.DataFrame:
    if "Log_Returns" in out_df.columns and "Log_Returns_raw" in out_df.columns:
        a = out_df["Log_Returns"].to_numpy(dtype=float)
        b = out_df["Log_Returns_raw"].to_numpy(dtype=float)
        if np.allclose(a, b, atol=1e-12, rtol=1e-9):
            out_df = out_df.drop(columns=["Log_Returns_raw"])
    return out_df

def make_unique_feature_names(base_names: List[str], existing: set) -> List[str]:
    out = []
    used = set(existing)
    for n in base_names:
        new = n if n not in used else f"{n}_scaled"
        k = 2
        while new in used:
            new = f"{n}_scaled{k}"
            k += 1
        out.append(new)
        used.add(new)
    return out

@dataclass
class ScalerSpec:
    mode: str  # "standard" | "minmax" | "hybrid"
    minmax_cols: Optional[List[str]]

class HybridScaler:
    def __init__(self, minmax_cols: List[str], all_feature_cols: List[str]):
        self.minmax_cols = list(minmax_cols)
        self.std_cols = [c for c in all_feature_cols if c not in self.minmax_cols]
        self.mm = MinMaxScaler()
        self.ss = StandardScaler()

    def fit(self, Xdf: pd.DataFrame):
        if self.minmax_cols:
            self.mm.fit(Xdf[self.minmax_cols].to_numpy(dtype=np.float32))
        if self.std_cols:
            self.ss.fit(Xdf[self.std_cols].to_numpy(dtype=np.float32))
        return self

    def transform(self, Xdf: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=Xdf.index)
        if self.minmax_cols:
            out[self.minmax_cols] = self.mm.transform(Xdf[self.minmax_cols].to_numpy(dtype=np.float32))
        if self.std_cols:
            out[self.std_cols]   = self.ss.transform(Xdf[self.std_cols].to_numpy(dtype=np.float32))
        return out[self.minmax_cols + self.std_cols] if self.minmax_cols else out[self.std_cols]

    def meta(self) -> Dict:
        return {"type": "hybrid", "minmax_cols": self.minmax_cols, "std_cols": self.std_cols}

def build_scaler(spec: ScalerSpec, feature_cols: List[str]):
    if spec.mode == "standard":
        return StandardScaler(), {"type": "standard"}
    if spec.mode == "minmax":
        return MinMaxScaler(), {"type": "minmax"}
    if spec.mode == "hybrid":
        if not spec.minmax_cols:
            raise ValueError("--minmax-cols must be provided for hybrid mode.")
        not_in = [c for c in spec.minmax_cols if c not in feature_cols]
        if not_in:
            raise ValueError(f"[ERROR] minmax_cols not in feature columns: {not_in}")
        hs = HybridScaler(spec.minmax_cols, feature_cols)
        return hs, hs.meta()
    raise ValueError(f"Unknown scaler mode: {spec.mode}")

def fit_pca_safely(X: np.ndarray, n_components_arg, max_pc: Optional[int] = None, random_state: int = 42) -> PCA:
    n_samples, n_features = X.shape
    if isinstance(n_components_arg, float):
        if not (0.0 < n_components_arg < 1.0):
            raise ValueError("When float, --pca-n-components must be in (0,1).")
        pca_tmp = PCA(n_components=n_components_arg, svd_solver="full", random_state=random_state).fit(X)
        k = int(pca_tmp.n_components_)
        if max_pc is not None and k > max_pc:
            return PCA(n_components=max_pc, svd_solver="full", random_state=random_state).fit(X)
        return pca_tmp
    else:
        k = int(n_components_arg)
        k = max(1, min(k, n_samples, n_features, max_pc if max_pc else k))
        return PCA(n_components=k, svd_solver="randomized", random_state=random_state).fit(X)

def ensure_dirs(model_type: str):
    out_root = os.path.join(FINAL_OUT_ROOT, model_type.lower())
    paths = {
        "train": os.path.join(out_root, "train"),
        "val":   os.path.join(out_root, "val"),
        "test":  os.path.join(out_root, "test"),
        "root":  out_root
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths

# -------------- PATCHED WRITERS --------------
def transform_and_save_fold(fd: Dict,
                            model_type: str,
                            feature_cols: List[str],
                            keep_cols: List[str],
                            scaler,
                            pca: Optional[PCA],
                            out_paths: Dict[str, str]) -> Dict:
    key_train = "train_path_lstm" if model_type == "lstm" else "train_path_arima"
    key_val   = "val_path_lstm"   if model_type == "lstm" else "val_path_arima"
    key_test  = "test_path_lstm"  if model_type == "lstm" else "test_path_arima"

    outputs = {}
    for split, rel in [("train", fd.get(key_train)), ("val", fd.get(key_val)), ("test", fd.get(key_test))]:
        if not rel:
            continue
        df = read_df(rel)
        if df is None or any(c not in df.columns for c in feature_cols):
            continue

        # Fit transforms on the given split (use fitted scaler/pca)
        if isinstance(scaler, (StandardScaler, MinMaxScaler)):
            Xs = scaler.transform(df[feature_cols].to_numpy(dtype=np.float32, copy=False))
        else:
            Xs = scaler.transform(df[feature_cols]).to_numpy(dtype=np.float32)

        out_df = df[ [c for c in keep_cols if c in df.columns] ].reset_index(drop=True).copy()

        if model_type.lower() == "arima" and feature_cols == ["Log_Returns"]:
            out_df["Log_Returns_scaled"] = Xs.ravel().astype(np.float32)
        else:
            if pca is None:
                Xout = Xs
                out_cols = feature_cols
            else:
                Xout = pca.transform(Xs)
                out_cols = [f"PC{i+1}" for i in range(Xout.shape[1])]
            out_cols = make_unique_feature_names(out_cols, set(out_df.columns))
            feat_df = pd.DataFrame(Xout, columns=out_cols)
            out_df = pd.concat([out_df, feat_df], axis=1)

        out_df = drop_duplicate_log_returns(out_df)
        
        out_file = os.path.join(out_paths[split], os.path.basename(absolute_path_from_rel(rel)))
        out_df.to_csv(out_file, index=False)
        outputs[split] = out_file

    updated = dict(fd)
    if "train" in outputs: updated["final_train_path"] = os.path.relpath(outputs["train"], OUTPUT_BASE_DIR)
    if "val"   in outputs: updated["final_val_path"]   = os.path.relpath(outputs["val"],   OUTPUT_BASE_DIR)
    if "test"  in outputs: updated["final_test_path"]  = os.path.relpath(outputs["test"],  OUTPUT_BASE_DIR)
    return updated

def transform_and_save_test_csv(test_csv_path: str,
                                model_type: str,
                                feature_cols: List[str],
                                keep_cols: List[str],
                                scaler,
                                pca: Optional[PCA],
                                out_paths: Dict[str, str]) -> str:
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"[ERROR] --test-csv not found: {test_csv_path}")
    df = pd.read_csv(test_csv_path)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[ERROR] Test CSV missing required feature columns: {missing}")

    if isinstance(scaler, (StandardScaler, MinMaxScaler)):
        Xs = scaler.transform(df[feature_cols].to_numpy(dtype=np.float32, copy=False))
    else:
        Xs = scaler.transform(df[feature_cols]).to_numpy(dtype=np.float32)

    out_df = df[ [c for c in keep_cols if c in df.columns] ].reset_index(drop=True).copy()

    if model_type.lower() == "arima" and feature_cols == ["Log_Returns"]:
        out_df["Log_Returns_scaled"] = Xs.ravel().astype(np.float32)
    else:
        if pca is None:
            Xout = Xs
            out_cols = feature_cols
        else:
            Xout = pca.transform(Xs)
            out_cols = [f"PC{i+1}" for i in range(Xout.shape[1])]
        out_cols = make_unique_feature_names(out_cols, set(out_df.columns))
        out_df = pd.concat([out_df, pd.DataFrame(Xout, columns=out_cols)], axis=1)

    out_df = drop_duplicate_log_returns(out_df)

    out_name = f"{model_type.lower()}_test_scaled.csv"
    out_file = os.path.join(out_paths["test"], out_name)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    out_df.to_csv(out_file, index=False)
    return out_file

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Final fold-level scaling + PCA on representative folds (LSTM/ARIMA)")
    ap.add_argument("--model-type", required=True, choices=["lstm", "arima"])
    ap.add_argument("--selected-folds-json", required=True,
                    help="Path to selected folds JSON (either list or report with 'selected_folds').")
    ap.add_argument("--folds-summary-path", default="",
                    help="Optional path to folds summary JSON; used to enrich selected folds if selected JSON lacks paths.")
    ap.add_argument("--feature-columns-path", default="",
                    help="Optional feature columns JSON; if missing will be inferred after enrichment.")
    ap.add_argument("--scaler-mode", default="standard", choices=["standard", "minmax", "hybrid"])
    ap.add_argument("--minmax-cols", default="", help="Comma-separated cols for MinMax (hybrid mode)")
    ap.add_argument("--pca-n-components", default="0.95", help="float (0,1) variance target or int")
    ap.add_argument("--pca-max-pc", type=int, default=64)

    # === PATCH #1: mutually exclusive flags for skipping PCA on ARIMA ===
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--skip-pca-for-arima", dest="skip_pca_for_arima", action="store_true",
                     help="Skip PCA for ARIMA if number of features <= 1 (default).")
    grp.add_argument("--no-skip-pca-for-arima", dest="skip_pca_for_arima", action="store_false",
                     help="Force PCA for ARIMA even if number of features <= 1.")
    ap.set_defaults(skip_pca_for_arima=True)

    ap.add_argument("--models-tag", default="")
    ap.add_argument("--assume-inputs-clean", action="store_true",
                    help="(reserved) if inputs are already checked.")
    ap.add_argument("--test-csv", default="",
                    help="Optional path to a single test CSV (not in folds). If provided, it will be transformed to <final>/<model>/test/<model>_test_scaled.csv")
    args = ap.parse_args()

    # 1) Load selected payload and normalize to list
    selected_payload = load_json(args.selected_folds_json)
    selected_list = resolve_selected_list(selected_payload)

    # 2) Enrich with paths (if missing)
    needs_enrich = not any(("train_path_lstm" in d or "train_path_arima" in d) for d in selected_list)
    if needs_enrich:
        if not args.folds_summary_path or not os.path.exists(args.folds_summary_path):
            raise RuntimeError("[ERROR] selected JSON has no paths. Please provide --folds-summary-path to enrich.")
        folds_summary = load_json(args.folds_summary_path)
        selected_list = enrich_selected_with_paths(selected_list, folds_summary, args.model_type)
        print(f"[INFO] Enriched {len(selected_list)} selected folds with original paths.")

    # 3) Load or infer feature columns
    if args.feature_columns_path and os.path.exists(args.feature_columns_path):
        feature_cols = load_json(args.feature_columns_path)
        if not isinstance(feature_cols, list) or not feature_cols:
            raise ValueError("[ERROR] Invalid feature columns JSON.")
        print(f"[INFO] Loaded feature columns from {args.feature_columns_path}: {len(feature_cols)} cols")
    else:
        print("[WARN] --feature-columns-path missing. Inferring from first TRAIN CSV...")
        feature_cols = infer_feature_columns_from_first_fold(selected_list, args.model_type)
        inferred_path = os.path.join(FINAL_OUT_ROOT, f"{args.model_type}_feature_columns_inferred.json")
        save_json(feature_cols, inferred_path)
        print(f"[INFO] Inferred {len(feature_cols)} features -> {inferred_path}")

    # 4) Fit scaler on union TRAIN
    print(f"[INFO] Selected folds: {len(selected_list)}")
    print(f"[INFO] Features: {len(feature_cols)}")
    union_train_df = gather_union_train(selected_list, args.model_type, feature_cols)

    minmax_cols = [c.strip() for c in args.minmax_cols.split(",") if c.strip()] if args.scaler_mode == "hybrid" else []
    scaler_obj, scaler_meta = build_scaler(ScalerSpec(args.scaler_mode, minmax_cols), feature_cols)

    if isinstance(scaler_obj, (StandardScaler, MinMaxScaler)):
        scaler_obj.fit(union_train_df.to_numpy(dtype=np.float32))
        Xs_union = scaler_obj.transform(union_train_df.to_numpy(dtype=np.float32))
    else:
        scaler_obj.fit(union_train_df)
        Xs_union = scaler_obj.transform(union_train_df).to_numpy(dtype=np.float32)

    # 5) Fit PCA
    try:
        pca_nc = float(args.pca_n_components) if "." in str(args.pca_n_components) else int(args.pca_n_components)
    except Exception:
        pca_nc = 0.95

    do_pca = True
    if args.model_type.lower() == "arima" and args.skip_pca_for_arima and Xs_union.shape[1] <= 1:
        do_pca = False

    if do_pca:
        pca = fit_pca_safely(Xs_union, n_components_arg=pca_nc, max_pc=args.pca_max_pc, random_state=42)
        reduced_dim = int(getattr(pca, "n_components_", Xs_union.shape[1]))
        evr = getattr(pca, "explained_variance_ratio_", None)
        cum_var = float(np.cumsum(evr)[-1]) if isinstance(evr, np.ndarray) and evr.size else None
    else:
        pca = None
        reduced_dim = Xs_union.shape[1]
        evr, cum_var = None, None
        print("[INFO] Skipping PCA for ARIMA (scale only).")

    # 6) Save models + meta
    tag = f"_{args.models_tag}" if args.models_tag else ""
    scaler_path = os.path.join(FINAL_MODELS_DIR, f"{args.model_type}_final_scaler{tag}.pkl")
    pca_path    = os.path.join(FINAL_MODELS_DIR, f"{args.model_type}_final_pca{tag}.pkl")
    dump(scaler_obj, scaler_path); dump(pca, pca_path)

    models_meta = {
        "model_type": args.model_type,
        "scaler": scaler_meta,
        "feature_cols": feature_cols,
        "pca": {
            "skipped": (pca is None),
            "n_components_param": None if pca is None else pca_nc,
            "reduced_dim": reduced_dim,
            "explained_variance_ratio": None if pca is None else (evr.tolist() if isinstance(evr, np.ndarray) else None),
            "cumulative_variance": cum_var
        },
        "fitted_on": "union_train_of_selected_folds",
        "num_selected_folds": len(selected_list)
    }
    meta_path = os.path.join(FINAL_MODELS_DIR, f"{args.model_type}_final_models_meta{tag}.json")
    save_json(models_meta, meta_path)
    print(f"[INFO] Saved FinalScaler -> {scaler_path}")
    print(f"[INFO] Saved FinalPCA    -> {pca_path}")
    print(f"[INFO] Saved meta        -> {meta_path}")
    print(f"[INFO] Final PCA reduced_dim={reduced_dim}, cum_var={cum_var}")

    # 7) Transform & save per fold
    out_dirs = ensure_dirs(args.model_type)
    keep_cols = NON_FEATURE_KEEP
    updated_folds = []
    for fd in selected_list:
        updated_folds.append(transform_and_save_fold(
            fd, args.model_type, feature_cols, keep_cols, scaler_obj, pca, out_dirs
        ))

    # 8) Transform single test CSV if provided
    if args.test_csv:
        try:
            out_test = transform_and_save_test_csv(
                args.test_csv, args.model_type, feature_cols, keep_cols, scaler_obj, pca, out_dirs
            )
            print(f"[INFO] Saved scaled TEST -> {out_test}")
        except Exception as e:
            print(f"[WARN] Failed to transform --test-csv: {e}")

    # 9) Save updated selected folds (final paths)
    base_name = os.path.splitext(os.path.basename(args.selected_folds_json))[0]
    out_json = os.path.join(FINAL_OUT_ROOT, args.model_type, f"{base_name}_final_paths.json")
    save_json(updated_folds, out_json)
    print(f"[INFO] Saved updated selected folds with final paths -> {out_json}")

if __name__ == "__main__":
    main()