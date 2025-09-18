import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

# ---------- helpers ----------
def load_json_any(p: Path):
    with p.open("r") as f:
        return json.load(f)

def unwrap_folds(obj):
    if isinstance(obj, dict):
        for k in ("arima","lstm","folds","data","results"):
            if k in obj and isinstance(obj[k], list):
                obj = obj[k]; break
    return obj

def pick_train_val_paths(rec: dict, mode: str) -> tuple[str|None, str|None]:
    train_keys = [k for k in rec if "train_path" in k]
    val_keys   = [k for k in rec if "val_path"   in k and "meta" not in k]
    def pref(keys):
        m = [k for k in keys if mode.lower() in k.lower()]
        return (m[0] if m else (keys[0] if keys else None))
    tk, vk = pref(train_keys), pref(val_keys)
    return (rec.get(tk) if tk else None, rec.get(vk) if vk else None)

def rwd_rmse(train_series: np.ndarray, val_series: np.ndarray) -> float:
    mu = float(np.mean(train_series))
    yhat = np.full_like(val_series, mu, dtype=float)
    return float(np.sqrt(np.mean((val_series - yhat)**2)))

def try_resolve(path_like: str|Path, search_roots: list[Path]) -> Path|None:
    p = Path(path_like)

    if p.is_absolute() and p.exists():
        return p

    for root in search_roots:
        cand = (root / p)
        if cand.exists():
            return cand
    return None

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("RWD RMSE baseline from final folds JSON (robust path resolution)")
    ap.add_argument("--t1-csv", required=True, help="Tier-1 results CSV (must have 'fold_id').")
    ap.add_argument("--folds-json", required=True, help="Final folds JSON with train/val paths.")
    ap.add_argument("--mode", required=True, choices=["arima","lstm"],
                    help="arima → default column Log_Returns; lstm → default column target.")
    ap.add_argument("--col", default="", help="Override column name if needed.")
    ap.add_argument("--out-csv", required=True, help="Output CSV with 'rmse_rwd'.")
    ap.add_argument("--debug", action="store_true", help="Verbose logging.")
    args = ap.parse_args()

    default_col = "Log_Returns" if args.mode == "arima" else "target"
    col = args.col.strip() or default_col

    t1 = pd.read_csv(args.t1_csv).copy()
    if "fold_id" not in t1.columns:
        raise ValueError("Tier-1 CSV must contain a 'fold_id' column.")
    folds_raw = unwrap_folds(load_json_any(Path(args.folds_json)))
    if not isinstance(folds_raw, list):
        raise ValueError("Folds JSON must unwrap to a list of fold records.")

    # map fold_id
    fold_index = {}
    for r in folds_raw:
        fid = r.get("global_fold_id", r.get("fold_id"))
        if fid is not None:
            fold_index[int(fid)] = r

    # build a robust set of search roots:
    json_path   = Path(args.folds_json).resolve()
    roots = [
        json_path.parent,                    
        json_path.parent.parent,              
        json_path.parent.parent.parent,       
        Path.cwd(),                          
        Path.cwd().parent                    
    ]

    roots = list(dict.fromkeys([r for r in roots if r is not None]))

    rmse_list, ok, miss = [], 0, 0
    for _, row in t1.iterrows():
        fid = int(row["fold_id"])
        rec = fold_index.get(fid)
        if rec is None:
            rmse_list.append(np.nan); miss += 1
            if args.debug: print(f"[WARN] fold_id {fid}: not in JSON")
            continue

        tr_rel, va_rel = pick_train_val_paths(rec, args.mode)
        if not tr_rel or not va_rel:
            rmse_list.append(np.nan); miss += 1
            if args.debug: print(f"[WARN] fold_id {fid}: no train/val path keys found")
            continue

        tr_path = try_resolve(tr_rel, roots)
        va_path = try_resolve(va_rel, roots)
        if args.debug:
            print(f"[DBG] fold {fid}: train={tr_rel} -> {tr_path}; val={va_rel} -> {va_path}")
        if tr_path == va_path:
            rmse_list.append(np.nan); miss += 1
            if args.debug: print(f"[WARN] fold_id {fid}: train and val paths are identical; skipping")
            continue

        if tr_path is None or va_path is None:
            rmse_list.append(np.nan); miss += 1
            if args.debug: print(f"[WARN] fold_id {fid}: cannot resolve CSV paths")
            continue

        try:
            tr = pd.read_csv(tr_path)
            va = pd.read_csv(va_path)
            use_col = col
            if use_col not in tr.columns or use_col not in va.columns:
                if args.mode == "arima" and "Log_Returns" in tr.columns and "Log_Returns" in va.columns:
                    use_col = "Log_Returns"
                elif args.mode == "lstm" and "target" in tr.columns and "target" in va.columns:
                    use_col = "target"
                else:
                    rmse_list.append(np.nan); miss += 1
                    if args.debug: print(f"[WARN] fold_id {fid}: column '{col}' not found")
                    continue

            trv = tr[use_col].astype(float).dropna().to_numpy()
            vav = va[use_col].astype(float).dropna().to_numpy()
            if trv.size == 0 or vav.size == 0:
                rmse_list.append(np.nan); miss += 1
                if args.debug: print(f"[WARN] fold_id {fid}: empty series after dropna")
                continue

            rmse_val = rwd_rmse(trv, vav)
            rmse_list.append(rmse_val); ok += 1

            if args.debug:
                print(f"[OK] fold {fid}: train={tr_path.name} ({trv.size} pts), "
                      f"val={va_path.name} ({vav.size} pts) → RMSE_RWD={rmse_val:.6f}")
        except Exception as e:
            rmse_list.append(np.nan); miss += 1
            if args.debug: print(f"[ERR ] fold_id {fid}: {e}")

    t1["rmse_rwd"] = rmse_list
    out_p = Path(args.out_csv); out_p.parent.mkdir(parents=True, exist_ok=True)
    t1.to_csv(out_p, index=False)
    print(f"[OK] wrote {out_p} — computed {ok} folds; {miss} missing")

if __name__ == "__main__":
    main()