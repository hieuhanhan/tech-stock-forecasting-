import warnings
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import argparse


# ---------- helpers ----------
def _skew(x, eps=1e-12):
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 3: return 0.0
    m = float(np.mean(x))
    s = float(np.std(x, ddof=1))
    if s < eps: return 0.0
    m3 = float(np.mean((x - m) ** 3))
    return m3 / (s ** 3 + eps)


def _fit_arima_resid_and_garch(history, h, p, q, scale_factor=1000.0, arch_rescale=False):
    """ARIMA(p,0,q) -> residuals, rồi GARCH(1,1) trên residuals -> var forecast; mean từ ARIMA."""
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore")
        arima_fit = ARIMA(history, order=(p, 0, q)).fit()
        resid = np.asarray(arima_fit.resid, dtype=float).ravel()

        garch = arch_model(resid * scale_factor, mean="Zero", vol="Garch", p=1, q=1,
                           dist="normal", rescale=arch_rescale)
        g_res = garch.fit(disp='off')

        f_var = np.asarray(g_res.forecast(horizon=h).variance.iloc[-1]).ravel()
        mu = np.asarray(arima_fit.forecast(steps=h)).ravel()
        var = np.maximum(f_var / (scale_factor ** 2 + 1e-12), 1e-12)
    return mu, var, resid


def _fit_arx_garch(history, h, p, scale_factor=1000.0, arch_rescale=False):
    """ARX(p) + GARCH(1,1) trực tiếp trên returns -> mean & var forecast."""
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore")
        hist_s = history * scale_factor
        model = arch_model(hist_s, mean="ARX", lags=p, vol="Garch", p=1, q=1,
                           dist="normal", rescale=arch_rescale)
        res = model.fit(disp='off')
        f = res.forecast(horizon=h)
        mu = np.asarray(f.mean.iloc[-1]).ravel()[:h] / (scale_factor + 1e-12)
        var = np.asarray(f.variance.iloc[-1]).ravel()[:h] / (scale_factor ** 2 + 1e-12)
        var = np.maximum(var, 1e-12)
    return mu, var


def compare_residuals_vs_returns(train_ret, val_ret,
                                 retrain_interval=20,
                                 p=3, q=1,
                                 thr_list=(0.01, 0.02, 0.05, 0.1, 0.2, 0.5),
                                 scale_factor=1000.0,
                                 arch_rescale=False,
                                 print_blocks=False):
    """
    Trả về:
      - stats_df: thống kê returns & residuals (mean/std/skew)
      - triggers_df: %trigger theo từng threshold cho 2 mode (residual vs return)
    """
    train = np.asarray(train_ret, dtype=float)
    val = np.asarray(val_ret, dtype=float)
    n = len(val)
    EPS = 1e-12

    # ---- Stats cơ bản ----
    returns_stats = dict(kind="returns",
                         mean=float(np.mean(train)),
                         std=float(np.std(train, ddof=1)),
                         skew=_skew(train))

    _, _, resid_all = _fit_arima_resid_and_garch(train, h=1, p=p, q=q,
                                                 scale_factor=scale_factor,
                                                 arch_rescale=arch_rescale)
    resid_stats = dict(kind="residuals",
                       mean=float(np.mean(resid_all)),
                       std=float(np.std(resid_all, ddof=1)),
                       skew=_skew(resid_all))
    stats_df = pd.DataFrame([returns_stats, resid_stats])

    # ---- Rolling blocks ----
    z_resid_all, z_return_all = [], []
    history = train.copy()

    for start in range(0, n, retrain_interval):
        end = min(start + retrain_interval, n)
        h = end - start
        if h <= 0: continue

        mu_r, var_r, _ = _fit_arima_resid_and_garch(history, h, p, q,
                                                    scale_factor=scale_factor,
                                                    arch_rescale=arch_rescale)
        z_r = mu_r / (np.sqrt(var_r) + EPS)
        z_resid_all.append(z_r)

        mu_x, var_x = _fit_arx_garch(history, h, p,
                                     scale_factor=scale_factor,
                                     arch_rescale=arch_rescale)
        z_x = mu_x / (np.sqrt(var_x) + EPS)
        z_return_all.append(z_x)

        if print_blocks:
            print(f"[BLK {start:04d}-{end:04d}] "
                  f"z_resid[min/mean/max]=({z_r.min():.4f}/{z_r.mean():.4f}/{z_r.max():.4f}) | "
                  f"z_return[min/mean/max]=({z_x.min():.4f}/{z_x.mean():.4f}/{z_x.max():.4f})")

        history = np.concatenate([history, val[start:end]])

    z_resid = np.concatenate(z_resid_all) if z_resid_all else np.array([])
    z_return = np.concatenate(z_return_all) if z_return_all else np.array([])

    rows = []
    for thr in thr_list:
        rows.append(dict(mode="residual_based", threshold=float(thr),
                         trigger_pct=float(np.mean(np.abs(z_resid) > thr) * 100.0 if z_resid.size else 0.0),
                         z_mean=float(z_resid.mean() if z_resid.size else 0.0),
                         z_std=float(z_resid.std(ddof=1) if z_resid.size else 0.0)))
        rows.append(dict(mode="return_based", threshold=float(thr),
                         trigger_pct=float(np.mean(np.abs(z_return) > thr) * 100.0 if z_return.size else 0.0),
                         z_mean=float(z_return.mean() if z_return.size else 0.0),
                         z_std=float(z_return.std(ddof=1) if z_return.size else 0.0)))
    triggers_df = pd.DataFrame(rows)

    return stats_df, triggers_df


def load_folds_meta(folds_path: Path):
    data = json.load(folds_path.open("r"))
    if isinstance(data, dict) and "arima" in data:
        return data["arima"]
    return data


def load_tier1_champions(tier1_path: Path):
    """
    Chấp nhận format bạn đưa: list các object {fold_id, best_params:{p,d,q}, ...}
    Trả về dict: {fold_id_int: {"p": p, "q": q}}
    """
    raw = json.load(tier1_path.open("r"))
    champions = {}
    if isinstance(raw, list):
        for item in raw:
            fid = int(item["fold_id"])
            bp = item.get("best_params", {})
            champions[fid] = {"p": int(bp.get("p", 1)), "q": int(bp.get("q", 1))}
    elif isinstance(raw, dict) and "results" in raw:
        # fallback nếu có format khác (tuỳ bạn)
        for item in raw["results"]:
            fid = int(item["fold_id"])
            bp = item.get("best_params", {})
            champions[fid] = {"p": int(bp.get("p", 1)), "q": int(bp.get("q", 1))}
    else:
        # Nếu là dict đơn giản {fid: {p,q}} thì parse luôn
        for k, v in raw.items():
            try:
                fid = int(k)
                champions[fid] = {"p": int(v["p"]), "q": int(v["q"])}
            except Exception:
                continue
    return champions


def main():
    ap = argparse.ArgumentParser("Compare residual-based vs return-based (standalone debug)")
    ap.add_argument("--folds-path", default="data/processed_folds/final/arima/arima_tuning_folds.json")
    ap.add_argument("--tier1-json", default="data/tuning_results/jsons/tier1_arima.json")
    ap.add_argument("--fold-id", type=int, default=None, help="Chọn fold cụ thể (mặc định: fold đầu tiên)")
    ap.add_argument("--retrain-interval", type=int, default=20)
    ap.add_argument("--scale-factor", type=float, default=1000.0)
    ap.add_argument("--arch-rescale", action="store_true")
    ap.add_argument("--print-blocks", action="store_true")
    ap.add_argument("--thr", type=str, default="0.01,0.02,0.05,0.1,0.2,0.5")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    folds_path = Path(args.folds_path).resolve()
    tier1_path = Path(args.tier1_json).resolve()

    folds = load_folds_meta(folds_path)
    champions = load_tier1_champions(tier1_path)

    if not folds:
        raise RuntimeError("Không load được danh sách folds.")

    # chọn fold
    if args.fold_id is not None:
        fold = next((f for f in folds if int(f["global_fold_id"]) == args.fold_id), None)
        if fold is None:
            raise RuntimeError(f"Fold {args.fold_id} không có trong meta.")
    else:
        fold = folds[0]

    fid = int(fold["global_fold_id"])
    if fid not in champions:
        raise RuntimeError(f"Fold {fid} không có trong Tier-1 champions.")

    p = int(champions[fid]["p"])
    q = int(champions[fid]["q"])
    logging.info(f"Fold {fid}: dùng Tier-1 best_params p={p}, q={q}")

    base_dir = folds_path.parents[2]

    # ưu tiên final_* nếu có
    train_rel = fold.get("final_train_path") or fold.get("train_path_arima")
    val_rel = fold.get("final_val_path") or fold.get("val_path_arima")
    if not train_rel or not val_rel:
        raise RuntimeError("Thiếu đường dẫn train/val trong meta.")

    train_csv = (base_dir / train_rel).resolve()
    val_csv = (base_dir / val_rel).resolve()
    if not (train_csv.exists() and val_csv.exists()):
        raise FileNotFoundError(f"Không tìm thấy CSV: {train_csv} hoặc {val_csv}")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    if "Log_Returns" not in train_df.columns or "Log_Returns" not in val_df.columns:
        raise RuntimeError("Thiếu cột Log_Returns trong CSV.")

    train_ret = train_df["Log_Returns"].dropna().to_numpy(float)
    val_ret = val_df["Log_Returns"].fillna(0).to_numpy(float)

    thr_list = tuple(float(x) for x in args.thr.split(",") if x.strip())

    stats_df, triggers_df = compare_residuals_vs_returns(
        train_ret, val_ret,
        retrain_interval=args.retrain_interval,
        p=p, q=q,
        thr_list=thr_list,
        scale_factor=args.scale_factor,
        arch_rescale=args.arch_rescale,
        print_blocks=args.print_blocks
    )

    print("\n=== Summary Stats ===")
    print(stats_df.to_string(index=False))
    print("\n=== Trigger Rates ===")
    print(triggers_df.to_string(index=False))


if __name__ == "__main__":
    main()