import os
import json
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

# ---------- IO roots ----------
OUTPUT_BASE_DIR = os.path.join("data", "processed_folds")
ARIMA_META_DIR = os.path.join(OUTPUT_BASE_DIR, "arima_meta")
LSTM_META_DIR  = os.path.join(OUTPUT_BASE_DIR, "lstm_meta")
os.makedirs(ARIMA_META_DIR, exist_ok=True)
os.makedirs(LSTM_META_DIR,  exist_ok=True)

# ---------- helpers ----------
def standardize_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        x = out[c].astype(float)
        mu = float(x.mean())
        sd = float(x.std()) or 1.0
        out[c] = (x - mu) / sd
    return out

def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32, copy=False)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    Y = X / norms
    S = Y @ Y.T
    return np.clip(S, 0.0, 1.0)

def jaccard_from_date_lists(a_list: List[str], b_list: List[str]) -> float:
    try:
        set_a, set_b = set(a_list), set(b_list)
        if not set_a and not set_b:
            return 0.0
        inter = len(set_a & set_b)
        union = len(set_a | set_b)
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0

def build_penalized_dissimilarity(meta_df: pd.DataFrame,
                                  base_sim: np.ndarray,
                                  lambda_overlap: float,
                                  gamma_same_ticker: float) -> np.ndarray:
    n = len(meta_df)
    dates = meta_df["date_list"].tolist() if "date_list" in meta_df.columns else [list() for _ in range(n)]
    tickers = meta_df["ticker"].fillna("UNK").tolist() if "ticker" in meta_df.columns else ["UNK"] * n

    overlap = np.zeros((n, n), dtype=np.float32)
    same_tkr = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        di = dates[i]
        for j in range(i + 1, n):
            dj = dates[j]
            o = jaccard_from_date_lists(di, dj)
            overlap[i, j] = overlap[j, i] = o
            if tickers[i] == tickers[j]:
                same_tkr[i, j] = same_tkr[j, i] = 1.0

    penalized_sim = base_sim - lambda_overlap * overlap - gamma_same_ticker * same_tkr
    penalized_sim = np.clip(penalized_sim, 0.0, 1.0)
    return 1.0 - penalized_sim

# ---------- coverage & stats ----------
def coverage_vector(sim: np.ndarray, selected: List[int]) -> np.ndarray:
    if not selected:
        return np.zeros(sim.shape[0], dtype=np.float32)
    return sim[:, selected].max(axis=1)


def coverage_stats(sim: np.ndarray, selected: List[int]) -> Dict[str, float]:
    cov = coverage_vector(sim, selected)
    if cov.size == 0:
        return {"mean": 0.0, "min": 0.0, "p05": 0.0, "p25": 0.0, "p50": 0.0, "p95": 0.0}
    return {
        "mean": float(np.mean(cov)),
        "min":  float(np.min(cov)),
        "p05":  float(np.quantile(cov, 0.05)),
        "p25":  float(np.quantile(cov, 0.25)),
        "p50":  float(np.quantile(cov, 0.50)),
        "p95":  float(np.quantile(cov, 0.95)),
    }

# ---------- selection core ----------
def _violates(idx: int,
              selected: List[int],
              tickers: List[str],
              dates: List[List[str]],
              per_ticker_max: int,
              hard_overlap: bool,
              max_overlap: float) -> bool:
    # per-ticker
    tkr = tickers[idx]
    if sum(1 for s in selected if tickers[s] == tkr) >= per_ticker_max:
        return True
    # hard overlap
    if hard_overlap and selected:
        di = dates[idx]
        for s in selected:
            if jaccard_from_date_lists(di, dates[s]) > max_overlap:
                return True
    return False

def fixed_k_facility_location(
    D: np.ndarray,
    meta_df: pd.DataFrame,
    k: int,
    hard_overlap: bool,
    max_overlap: float,
    per_ticker_max: int,
    tie_break: str = "diversity",
    rng: Optional[np.random.Generator] = None,
    gain_eps: float = 1e-12,
) -> List[int]:
    n = D.shape[0]
    sim = 1.0 - D
    dates  = meta_df["date_list"].tolist() if "date_list" in meta_df.columns else [list() for _ in range(n)]
    tickers = meta_df["ticker"].fillna("UNK").tolist() if "ticker" in meta_df.columns else ["UNK"] * n

    selected: List[int] = []
    covered = np.zeros(n, dtype=np.float32)

    order = list(range(n))
    if rng is None:
        rng = np.random.default_rng()
    rng.shuffle(order)

    while len(selected) < min(k, n):
        best_gain, best_idx = -1e18, None
        tied: List[int] = []

        for i in order:
            if i in selected:
                continue
            if _violates(i, selected, tickers, dates, per_ticker_max, hard_overlap, max_overlap):
                continue
            gain = float(np.maximum(covered, sim[:, i]).sum() - covered.sum())
            if gain > best_gain + gain_eps:
                best_gain, best_idx = gain, i
                tied = [i]
            elif abs(gain - best_gain) <= gain_eps:
                tied.append(i)
        if best_idx is None:
            break
        pick = best_idx
        if len(tied) > 1:
            if tie_break == "random":
                pick = rng.choice(tied)
            elif tie_break == "diversity" and selected:
                # choose i maximizing avg dissimilarity to current selected
                ds = D[np.ix_(tied, selected)].mean(axis=1)
                pick = tied[int(np.argmax(ds))]
            else:
                pick = tied[0]

        selected.append(pick)
        covered = np.maximum(covered, sim[:, pick])
    return selected

def adaptive_facility_location(
    D: np.ndarray,
    meta_df: pd.DataFrame,
    k_min: int,
    k_max: int,
    hard_overlap: bool,
    max_overlap: float,
    per_ticker_max: int,
    coverage_target: float,
    min_gain: float,
    elbow_tol: float,
    tie_break: str = "diversity",
    rng: Optional[np.random.Generator] = None,
    gain_eps: float = 1e-12,
) -> Dict:
    n = D.shape[0]
    sim = 1.0 - D
    dates  = meta_df["date_list"].tolist() if "date_list" in meta_df.columns else [list() for _ in range(n)]
    tickers = meta_df["ticker"].fillna("UNK").tolist() if "ticker" in meta_df.columns else ["UNK"] * n

    selected: List[int] = []
    covered = np.zeros(n, dtype=np.float32)
    coverage_curve: List[float] = []
    gain_curve: List[float] = []
    first_gain: Optional[float] = None

    if rng is None:
        rng = np.random.default_rng()
    order = list(range(n))
    rng.shuffle(order)

    for t in range(1, min(k_max, n) + 1):
        best_gain, best_idx = -1e18, None
        tied: List[int] = []
        for i in order:
            if i in selected:
                continue
            if _violates(i, selected, tickers, dates, per_ticker_max, hard_overlap, max_overlap):
                continue
            gain = float(np.maximum(covered, sim[:, i]).sum() - covered.sum())
            if gain > best_gain + gain_eps:
                best_gain, best_idx = gain, i
                tied = [i]
            elif abs(gain - best_gain) <= gain_eps:
                tied.append(i)
        if best_idx is None:
            break

        # tie-break
        pick = best_idx
        if len(tied) > 1:
            if tie_break == "random":
                pick = rng.choice(tied)
            elif tie_break == "diversity" and selected:
                ds = D[np.ix_(tied, selected)].mean(axis=1)
                pick = tied[int(np.argmax(ds))]
            else:
                pick = tied[0]

        selected.append(pick)
        covered = np.maximum(covered, sim[:, pick])
        mean_cov = float(covered.mean())
        coverage_curve.append(mean_cov)
        gain_curve.append(float(best_gain))
        if first_gain is None:
            first_gain = max(1e-12, float(best_gain))

        if t >= k_min:
            stop_by_cov   = coverage_target > 0 and mean_cov >= coverage_target
            stop_by_gain  = min_gain > 0 and best_gain <= min_gain
            stop_by_elbow = elbow_tol > 0 and (best_gain / first_gain) <= elbow_tol
            if stop_by_cov or stop_by_gain or stop_by_elbow:
                break

    return {"indices": selected, "coverage_curve": coverage_curve, "gain_curve": gain_curve}


# ---------- local search (1-swap) ----------
def one_swap_improve(
    D: np.ndarray,
    meta_df: pd.DataFrame,
    sel_idx: List[int],
    per_ticker_max: int,
    hard_overlap: bool,
    max_overlap: float,
    max_iter: int = 50,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[int], float]:
    n = D.shape[0]
    sim = 1.0 - D
    tickers = meta_df["ticker"].fillna("UNK").tolist()
    dates  = meta_df["date_list"].tolist() if "date_list" in meta_df.columns else [list() for _ in range(n)]

    if rng is None:
        rng = np.random.default_rng()

    def violates_set(selected: List[int]) -> bool:
        # per-ticker counts
        counts = {}
        for s in selected:
            t = tickers[s]
            counts[t] = counts.get(t, 0) + 1
            if counts[t] > per_ticker_max:
                return True
        if hard_overlap:
            for i in range(len(selected)):
                for j in range(i+1, len(selected)):
                    if jaccard_from_date_lists(dates[selected[i]], dates[selected[j]]) > max_overlap:
                        return True
        return False

    def cov_mean(selected: List[int]) -> float:
        if not selected:
            return 0.0
        return float((sim[:, selected].max(axis=1)).mean())

    best = sel_idx[:]
    best_cov = cov_mean(best)

    for _ in range(max_iter):
        improved = False
        # try random order to escape deterministic traps
        outs = best[:]
        rng.shuffle(outs)
        for out in outs:
            candidates = [i for i in range(n) if i not in best]
            rng.shuffle(candidates)
            for inn in candidates:
                trial = best[:]
                trial.remove(out)
                trial.append(inn)
                if violates_set(trial):
                    continue
                cov = cov_mean(trial)
                if cov > best_cov + 1e-9:
                    best, best_cov = trial, cov
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best, best_cov


# ---------- multistart wrappers ----------
def run_fixed_k_multistart(
    D: np.ndarray,
    meta_df: pd.DataFrame,
    k: int,
    hard_overlap: bool,
    max_overlap: float,
    per_ticker_max: int,
    tie_break: str,
    multistart: int,
    base_seed: Optional[int] = None,
    local_search: bool = False,
    swap_iters: int = 50,
) -> Tuple[List[int], Dict[str, float]]:
    sim = 1.0 - D
    best_sel: List[int] = []
    best_stats: Dict[str, float] = {"mean": -1.0}

    for s in range(multistart):
        rng = np.random.default_rng(None if base_seed is None else base_seed + s)
        sel = fixed_k_facility_location(
            D, meta_df, k, hard_overlap, max_overlap, per_ticker_max, tie_break=tie_break, rng=rng
        )
        if local_search and sel:
            sel, _ = one_swap_improve(D, meta_df, sel, per_ticker_max, hard_overlap, max_overlap, max_iter=swap_iters, rng=rng)
        stats = coverage_stats(sim, sel)
        if (stats["mean"] > best_stats.get("mean", -1)) or (
            abs(stats["mean"] - best_stats.get("mean", -1)) <= 1e-12 and len(sel) > len(best_sel)
        ):
            best_sel, best_stats = sel, stats
    return best_sel, best_stats

# ---------- main ----------
def main():
    p = argparse.ArgumentParser(description="Select representative folds with optional enrichment of paths.")
    p.add_argument("--model_type", required=True, choices=["lstm", "arima"])
    p.add_argument("--meta_json_path", required=True, help="Meta-features JSON")

    # evaluation mode (print only)
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--k_min", type=int, default=10)
    p.add_argument("--k_max", type=int, default=40)
    p.add_argument("--k_step", type=int, default=5)

    # selection modes
    p.add_argument("--adaptive_k", action="store_true")
    p.add_argument("--k", type=int, default=40)

    # penalties & constraints
    p.add_argument("--lambda_overlap", type=float, default=0.9)
    p.add_argument("--gamma_same_ticker", type=float, default=0.25)
    p.add_argument("--hard_overlap", action="store_true")
    p.add_argument("--max_overlap", type=float, default=0.7)
    p.add_argument("--per_ticker_max", type=int, default=2)

    # adaptive stopping
    p.add_argument("--coverage_target", type=float, default=0.0)
    p.add_argument("--min_gain", type=float, default=0.0)
    p.add_argument("--elbow_tol", type=float, default=0.0)

    # NEW: quality/speed knobs
    p.add_argument("--multistart", type=int, default=1, help="Number of random restarts for greedy")
    p.add_argument("--tie_break", choices=["diversity", "random", "first"], default="diversity")
    p.add_argument("--local_search", action="store_true", help="Enable 1-swap local search refinement")
    p.add_argument("--swap_iters", type=int, default=50, help="Max iterations for 1-swap local search")
    p.add_argument("--seed", type=int, default=None, help="Base RNG seed for reproducibility")

    # outputs
    p.add_argument("--out_selected_json", type=str, default="")
    p.add_argument("--out_report_json", type=str, default="")
    args = p.parse_args()

    # load meta
    with open(args.meta_json_path, "r") as f:
        meta_rows = json.load(f)
    meta_df = pd.DataFrame(meta_rows)

    # numeric cols
    exclude = {"global_fold_id", "ticker", "date_min", "date_max", "date_list"}
    num_cols = [c for c in meta_df.columns if c not in exclude and np.issubdtype(meta_df[c].dtype, np.number)]
    if not num_cols:
        raise RuntimeError("[ERROR] No numeric meta-features available.")

    X = standardize_columns(meta_df[num_cols], num_cols).values.astype(np.float32, copy=False)
    base_sim = cosine_similarity_matrix(X)
    D = build_penalized_dissimilarity(
        meta_df=meta_df,
        base_sim=base_sim,
        lambda_overlap=float(args.lambda_overlap),
        gamma_same_ticker=float(args.gamma_same_ticker),
    )

    # evaluation only
    if args.eval_only:
        print(f"[INFO] EVALUATE k in [{args.k_min}, {args.k_max}] step {args.k_step} | "
              f"hard_overlap={bool(args.hard_overlap)} max_overlap={args.max_overlap} per_ticker_max={args.per_ticker_max}")
        sim = 1.0 - D
        for k in range(int(args.k_min), int(args.k_max) + 1, int(args.k_step)):
            sel, stats = run_fixed_k_multistart(
                D, meta_df, k,
                bool(args.hard_overlap), float(args.max_overlap), int(args.per_ticker_max),
                tie_break=args.tie_break, multistart=max(1, int(args.multistart)), base_seed=args.seed,
                local_search=bool(args.local_search), swap_iters=int(args.swap_iters)
            )
            diversity = 0.0
            if len(sel) > 1:
                subD = D[np.ix_(sel, sel)]
                tri = subD[np.triu_indices(len(sel), 1)]
                diversity = float(tri.mean()) if tri.size else 0.0
            print(
                f"k={k:>3} | cov_mean={stats['mean']:.3f} min={stats['min']:.3f} p05={stats['p05']:.3f} p95={stats['p95']:.3f} | "
                f"diversity_mean={diversity:.3f} | selected={len(sel)}"
            )
        return

    # selection
    if args.adaptive_k:
        rng = np.random.default_rng(args.seed)
        res = adaptive_facility_location(
            D, meta_df,
            int(args.k_min), int(args.k_max),
            bool(args.hard_overlap), float(args.max_overlap), int(args.per_ticker_max),
            float(args.coverage_target), float(args.min_gain), float(args.elbow_tol),
            tie_break=args.tie_break, rng=rng
        )
        sel_idx = res["indices"]
        coverage_curve = res["coverage_curve"]
        gain_curve = res["gain_curve"]
        # optional local search after adaptive
        if args.local_search and sel_idx:
            sel_idx, _ = one_swap_improve(D, meta_df, sel_idx, int(args.per_ticker_max), bool(args.hard_overlap), float(args.max_overlap), int(args.swap_iters), rng)
    else:
        sel_idx, stats_best = run_fixed_k_multistart(
            D, meta_df, int(args.k),
            bool(args.hard_overlap), float(args.max_overlap), int(args.per_ticker_max),
            tie_break=args.tie_break, multistart=max(1, int(args.multistart)), base_seed=args.seed,
            local_search=bool(args.local_search), swap_iters=int(args.swap_iters)
        )
        # build simple curves from fixed picks
        sim = 1.0 - D
        coverage_curve, gain_curve = [], []
        covered = np.zeros(D.shape[0], dtype=np.float32)
        for idx in sel_idx:
            gain = float(np.maximum(covered, sim[:, idx]).sum() - covered.sum())
            covered = np.maximum(covered, sim[:, idx])
            coverage_curve.append(float(covered.mean()))
            gain_curve.append(gain)

    # metrics
    sim = 1.0 - D
    cov_stats = coverage_stats(sim, sel_idx)
    if len(sel_idx) > 1:
        subD = D[np.ix_(sel_idx, sel_idx)]
        tri = subD[np.triu_indices(len(sel_idx), 1)]
        diversity = float(tri.mean()) if tri.size else 0.0
    else:
        diversity = 0.0

    selected_simple = []
    for rank, i in enumerate(sel_idx, 1):
        row = meta_rows[i]
        selected_simple.append({
            "rank": rank,
            "global_fold_id": row.get("global_fold_id"),
            "ticker": row.get("ticker", "UNK"),
            "date_min": row.get("date_min", ""),
            "date_max": row.get("date_max", "")
        })

    result = {
        "mode": "adaptive" if args.adaptive_k else "fixed",
        "k": len(sel_idx),
        "metrics": {
            "coverage_mean": cov_stats["mean"],
            "coverage_min":  cov_stats["min"],
            "coverage_p05":  cov_stats["p05"],
            "coverage_p25":  cov_stats["p25"],
            "coverage_p50":  cov_stats["p50"],
            "coverage_p95":  cov_stats["p95"],
            "diversity_mean": diversity
        },
        "curves": {
            "coverage": coverage_curve,
            "gain": gain_curve
        },
        "selected_folds": selected_simple,
        "params": {
            "lambda_overlap": float(args.lambda_overlap),
            "gamma_same_ticker": float(args.gamma_same_ticker),
            "hard_overlap": bool(args.hard_overlap),
            "max_overlap": float(args.max_overlap),
            "per_ticker_max": int(args.per_ticker_max),
            "k_fixed": int(args.k),
            "k_min": int(args.k_min),
            "k_max": int(args.k_max),
            "coverage_target": float(args.coverage_target),
            "min_gain": float(args.min_gain),
            "elbow_tol": float(args.elbow_tol),
            "multistart": int(args.multistart),
            "tie_break": args.tie_break,
            "local_search": bool(args.local_search),
            "swap_iters": int(args.swap_iters),
            "seed": args.seed,
        }
    }

    # output path
    if not args.out_selected_json:
        base = "selected_lstm.json" if args.model_type.lower() == "lstm" else "selected_arima.json"
        out_path = os.path.join(LSTM_META_DIR if args.model_type.lower()=="lstm" else ARIMA_META_DIR, base)
    else:
        out_path = args.out_selected_json

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[INFO] Saved selected folds -> {out_path}")

    if args.out_report_json:
        with open(args.out_report_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[INFO] Saved diagnostics -> {args.out_report_json}")


if __name__ == "__main__":
    main()