import warnings
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from typing import List, Dict, Any, Union
import logging
from arch import arch_model

# Silence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Non-invertible")
warnings.filterwarnings("ignore", message="Maximum Likelihood optimization failed")
warnings.filterwarnings("ignore", message="Too few observations")

# === Constants ===
LOG_DIR = "logs"
GA_POP_SIZE = 35
GA_N_GEN = 25
BO_N_CALLS = 50
TOP_N_SEEDS = 5
BASE_COST = 0.0005
SLIPPAGE = 0.0002
DEFAULT_T2_NGEN_ARIMA = 40
DEFAULT_MIN_BLOCK_VOL = 0.0015

# -------- Metrics --------
def sharpe_ratio(simple_returns):
    r = np.asarray(simple_returns, dtype=float)
    if r.size < 2:
        return 0.0
    std = float(np.std(r, ddof=1))
    return 0.0 if std == 0.0 else (float(np.mean(r)) / std) * np.sqrt(252.0)

def max_drawdown(data, input_type="simple"):
    x = np.asarray(data, dtype=float)
    if x.size == 0:
        return 0.0
    if input_type == "price":
        C = x / (x[0] + np.finfo(float).eps)
    elif input_type == "simple":
        C = np.cumprod(1.0 + x)
    elif input_type == "log":
        C = np.exp(np.cumsum(x))
    elif input_type == "auto":
        C = np.exp(np.cumsum(x)) if np.min(x) <= -1.0 else np.cumprod(1.0 + x)
    else:
        raise ValueError("input_type must be one of {'price','simple','log','auto'}")
    peak = np.maximum.accumulate(C)
    dd = (peak - C) / (peak + np.finfo(float).eps)
    return float(np.max(dd))

eval_cache = {}
def evaluate_strategy(p, q, thr, returns, residuals, vol_series, mode="block", round_thr=6):
    key = (p, q, round(thr, round_thr), mode)

    if key in eval_cache:
        return eval_cache[key]

    signals = []
    for i in range(len(returns)):
        z = residuals[i] / (vol_series[i] + 1e-8)
        if mode == "block":
            adaptive_thr = thr * (vol_series[i] / (np.mean(vol_series[max(0, i-20):i+1]) + 1e-8))
        else:
            adaptive_thr = thr

        signals.append(int(abs(z) > adaptive_thr))

    signals = np.array(signals)
    strat_ret = returns * signals

    sharpe = np.mean(strat_ret) / (np.std(strat_ret) + 1e-8)
    mdd = np.min(np.cumsum(strat_ret) - np.maximum.accumulate(np.cumsum(strat_ret)))

    eval_cache[key] = (sharpe, mdd)
    return sharpe, mdd

# -------- Tier-1 --------
def arima_rmse(params, train, val, normalize: bool = False) -> float:
    p, q = map(int, params)
    p = int(np.clip(p, 1, 7))
    q = int(np.clip(q, 1, 7))
    try:
        model = ARIMA(train, order=(p, 0, q)).fit()
        forecast = model.forecast(steps=len(val))
        rmse = sqrt(mean_squared_error(val, forecast))
        if normalize:
            denom = float(np.std(val)) or 1.0
            return float(rmse / denom)
        return float(rmse)
    except Exception:
        return 1e3

# Tier-2: Periodic ARIMA Strategy Objective
def create_periodic_arima_objective(
    train: np.ndarray,
    val: np.ndarray,
    retrain_interval: int,
    cost_per_trade: float,
    min_block_vol: float = DEFAULT_MIN_BLOCK_VOL,
    scale_factor: float = 1000.0,
    arch_rescale_flag: bool = False,
    debug: bool = True,
    # --- adaptive config ---
    adaptive_mode: str = "block",     # "block" | "power" | "quantile"
    soft_factor: float = 0.2,         # "block"
    alpha: float = -0.08,             # "power"  (nhẹ hơn: -0.08 thay vì -0.2)
    ref_vol_ema_span: int = 60,       # "power"
    q_window: int = 120,              # "quantile" (buffer |z| gần nhất)
    q_level: float = 0.80,            # "quantile" (hạ còn 0.80 để tránh nghẹt lệnh)
    use_arima_residuals: bool = True,
    thr_min: float = 0.01,
    thr_max: float = 0.30,
):
   
    train = np.asarray(train, dtype=float)
    val = np.asarray(val, dtype=float)

    # --- penalties & guards ---
    W_THR_OUT  = 10.0
    W_FIT_FAIL = 5.0
    W_LOW_VOL  = 2.0
    W_NO_TRADE = 0.5
    W_LOW_ACT  = 2.0     # ít block kích hoạt
    W_FEW_TR   = 5.0     # quá ít trades

    # --- clamps ---
    RATIO_LO, RATIO_HI = 0.5, 2.0     # clamp ratio vol/ref_vol ở power
    THR_CAP            = 0.25         # nắp trên thr hiệu dụng để tránh nghẹt lệnh

    EPS = 1e-8

    def _skew(x: np.ndarray, eps=1e-12):
        x = np.asarray(x, dtype=float)
        if x.size < 3:
            return 0.0
        m = float(np.mean(x))
        s = float(np.std(x, ddof=1))
        if s < eps:
            return 0.0
        m3 = float(np.mean((x - m) ** 3))
        return float(m3 / (s**3 + eps))
    
    def objective(x):
        p = int(np.clip(np.rint(x[0]), 1, 7))
        q = int(np.clip(np.rint(x[1]), 1, 7))
        thr = float(x[2])

        penalty = 0.0
        if thr < thr_min:
            penalty += W_THR_OUT * (thr_min - thr)
            thr = thr_min
        elif thr > thr_max:
            penalty += W_THR_OUT * (thr - thr_max)
            thr = thr_max

        # init
        history = train.copy()
        simple_all, log_all = [], []
        total_trades, prev_pos_last = 0, 0

        # --- state cho "power" & "quantile"
        if adaptive_mode == "power":
            ref_vol = max(float(np.std(train)), min_block_vol)   # mốc ban đầu
            ema_k   = 2.0 / (float(ref_vol_ema_span) + 1.0)
        else:
            ref_vol, ema_k = None, None

        if adaptive_mode == "quantile":
            z_hist_abs: list[float] = []  # buffer |z| không leak
        else:
            z_hist_abs = None                     

        total_blocks, trig_blocks = 0, 0
        z_debug = []
        pos_all = []
        n = len(val)
        
        for start in range(0, n, retrain_interval):
            end = min(start + retrain_interval, n)
            block_log = val[start:end]
            h = end - start
            if h <= 0:
                continue
            total_blocks += 1

            block_vol = float(np.std(block_log))

            # update EMA ref_vol
            if adaptive_mode == "power":
                ref_vol = (1.0 - ema_k) * ref_vol + ema_k * max(block_vol, EPS)

            # --- Skip block nếu quá yên ắng ---
            if block_vol < float(min_block_vol):
                penalty += W_LOW_VOL * (min_block_vol - block_vol)
                history = np.concatenate([history, block_log])
                if debug:
                    logging.debug(f"[BLK {start:04d}-{end:04d}] SKIP vol={block_vol:.3e} < {min_block_vol:.3e}")
                continue
            
            # --- Fit ARIMA -> residual, rồi GARCH(resid) ---
            try:
                arima_fit = ARIMA(history, order=(p, 0, q)).fit()
                resid = np.asarray(arima_fit.resid, dtype=float).ravel()

                garch = arch_model(resid * scale_factor, mean="Zero", vol="Garch",
                                   p=1, q=1, dist="normal", rescale=arch_rescale_flag)
                g_res = garch.fit(disp='off')

                f_var = np.asarray(g_res.forecast(horizon=h).variance.iloc[-1]).ravel()
                forecast_mean = np.asarray(arima_fit.forecast(steps=h)).ravel()
                forecast_var  = np.maximum(f_var / (scale_factor**2 + EPS), EPS)

                if debug:
                    r_mean = float(np.mean(resid))
                    r_std  = float(np.std(resid, ddof=1))
                    r_skew = _skew(resid)
            except Exception as e:
                logging.debug(f"[FIT_FAIL] p={p}, q={q}, err={e}")
                penalty += W_FIT_FAIL
                history = np.concatenate([history, block_log])
                continue

            # --- z-score ---
            z = forecast_mean / (np.sqrt(forecast_var) + EPS)
            z_debug.extend(z.tolist())

            # --- tính adaptive threshold theo mode ---
            if adaptive_mode == "block":
                adaptive_thr = thr * (1.0 + soft_factor * (block_vol / max(min_block_vol, EPS)))
                adaptive_thr = float(np.clip(adaptive_thr, thr_min, THR_CAP))
            elif adaptive_mode == "power":
                ratio = max(block_vol, EPS) / max(ref_vol, EPS)
                ratio = float(np.clip(ratio, RATIO_LO, RATIO_HI))
                adaptive_thr = thr * (ratio ** alpha)
                adaptive_thr = float(np.clip(adaptive_thr, thr_min, THR_CAP))
            elif adaptive_mode == "quantile":
                have_warmup = (z_hist_abs is not None) and (len(z_hist_abs) >= max(20, int(0.25*q_window)))
                if have_warmup:
                    thr_q = float(np.quantile(np.asarray(z_hist_abs), q_level))
                    adaptive_thr = float(np.clip(thr_q, thr_min, thr_max))
                else:
                    adaptive_thr = thr
            else:
                adaptive_thr = thr

             # --- soft rebalance / tín hiệu vị thế ---
            pos_now = np.zeros(h, dtype=int)
            prev_pos, triggers = prev_pos_last, 0
            for t in range(h):
                if abs(z[t]) > adaptive_thr:
                    new_pos = 1 if z[t] > 0 else -1
                    triggers += 1
                else:
                    new_pos = prev_pos
                pos_now[t] = new_pos
                prev_pos = new_pos

            if triggers > 0:
                trig_blocks += 1
            pos_all.extend(pos_now.tolist())

            # --- block log ---
            if debug:
                zmin, zmax, zmean = float(np.min(z)), float(np.max(z)), float(np.mean(z))
                if adaptive_mode == "block":
                    logging.debug(
                        f"[BLK {start:04d}-{end:04d}] mode=block | p={p},q={q},thr={thr:.3f},thr_eff={adaptive_thr:.3f} | "
                        f"vol={block_vol:.3e}, trig={triggers}/{h} | "
                        f"z[{zmin:.2f}/{zmean:.2f}/{zmax:.2f}] | "
                        f"resid[{r_mean:.2e},{r_std:.2e},{r_skew:.2f}]"
                    )
                elif adaptive_mode == "power":
                    logging.debug(
                        f"[BLK {start:04d}-{end:04d}] mode=power | p={p},q={q},thr={thr:.3f},thr_eff={adaptive_thr:.3f} | "
                        f"vol={block_vol:.3e}, ref_vol={ref_vol:.3e}, alpha={alpha:.3f}, trig={triggers}/{h} | "
                        f"z[{zmin:.2f}/{zmean:.2f}/{zmax:.2f}] | "
                        f"resid[{r_mean:.2e},{r_std:.2e},{r_skew:.2f}]"
                    )
                elif adaptive_mode == "quantile":
                    used_q = int(round(q_level * 100))
                    logging.debug(
                        f"[BLK {start:04d}-{end:04d}] mode=quantile | p={p},q={q},thr_base={thr:.3f},thr_q{used_q}={adaptive_thr:.3f} | "
                        f"vol={block_vol:.3e}, hist_n={len(z_hist_abs) if z_hist_abs is not None else 0}, trig={triggers}/{h} | "
                        f"z[{zmin:.2f}/{zmean:.2f}/{zmax:.2f}] | "
                        f"resid[{r_mean:.2e},{r_std:.2e},{r_skew:.2f}]"
                    )

            # cập nhật buffer quantile sau khi dùng ngưỡng (không leakage)
            if adaptive_mode == "quantile":
                z_hist_abs.extend(np.abs(z).tolist())
                if len(z_hist_abs) > q_window:
                    z_hist_abs = z_hist_abs[-q_window:]

            # --- PnL ---
            signals_full = np.concatenate([[prev_pos_last], pos_now])
            trades = np.abs(np.diff(signals_full))
            cost_vec = cost_per_trade * trades
            total_trades += int(trades.sum())

            block_simple = np.exp(block_log) - 1.0
            ret_simple = block_simple * pos_now - cost_vec
            ret_simple = np.clip(ret_simple, -0.9999, None)
            ret_log = np.log1p(ret_simple)

            simple_all.append(ret_simple)
            log_all.append(ret_log)

            history = np.concatenate([history, block_log])
            prev_pos_last = pos_now[-1]

        # === guard nếu không có block hợp lệ ===
        if not simple_all:
            penalty += W_FIT_FAIL
            if debug:
                logging.debug(f"[KPI] blocks={total_blocks}, trig_blocks={trig_blocks}, trades={total_trades} (NO_VALID_BLOCKS)")
            return 1.0 + penalty, 1.0 + penalty

        net_simple = np.concatenate(simple_all)
        net_log    = np.concatenate(log_all)

        # --- hoạt động & penalty mềm để loại nghiệm “đứng im” ---
        if total_trades == 0:
            penalty += W_NO_TRADE
        act_ratio = (trig_blocks / total_blocks) if total_blocks > 0 else 0.0
        if act_ratio < 0.30:     # <30% block có trigger
            penalty += W_LOW_ACT
        if total_trades < 3:
            penalty += W_FEW_TR

        raw_sharpe = sharpe_ratio(net_simple)
        raw_mdd = max_drawdown(net_log, input_type="log")

        f0 = -raw_sharpe
        f1 = raw_mdd
        if not np.isfinite(f0):
            penalty += W_FIT_FAIL
            f0 = 1.0
        if not np.isfinite(f1):
            penalty += W_FIT_FAIL
            f1 = 1.0
                # === Fold-level debug ===
        pos_all = np.array(pos_all, dtype=int)
        sig_hash = hash(tuple(pos_all.tolist()))
        if debug:
            logging.debug(f"[SIG] p={p},q={q},thr={thr:.4f} hash={sig_hash} trades={total_trades}")
            logging.debug(f"[KPI] blocks={total_blocks}, trig_blocks={trig_blocks}, "
                          f"trades={total_trades}, Sharpe={raw_sharpe:.3f}, MDD={raw_mdd:.3f}, act={act_ratio:.2%}")

            if not hasattr(objective, "_last_runs"):
                objective._last_runs = []
            objective._last_runs.append({
                "p": p, "q": q, "thr": thr,
                "z_path": np.array(z_debug, dtype=float),
                "raw_sharpe": float(raw_sharpe),
                "raw_mdd": float(raw_mdd),
                "penalty": float(penalty),
                "total_trades": int(total_trades),
                "act_ratio": float(act_ratio),
            })
            if total_blocks > 0:
                logging.debug(
                    f"[SUMMARY] blocks={total_blocks}, trig_blocks={trig_blocks} ({100.0*act_ratio:.1f}%), "
                    f"trades={total_trades}, Sharpe={raw_sharpe:.3f}, MDD={raw_mdd:.3f}, Penalty={penalty:.3f}"
                )

        return float(f0 + penalty), float(f1 + penalty)

    return objective

def backtest_strategy(returns, residuals, p, q, thr, vol_series, mode="adaptive"):
    signals = []
    adaptive_thr_list = []

    for i in range(len(returns)):
        # z-score theo residual hoặc raw return
        z = residuals[i] / (vol_series[i] + 1e-8)

        # adaptive threshold
        if mode == "adaptive":
            block_vol = vol_series[i]
            ref_vol   = np.mean(vol_series[max(0,i-20):i+1])  # EMA hoặc rolling
            adaptive_thr = thr * (block_vol / (ref_vol + 1e-8))
        else:
            adaptive_thr = thr

        signals.append(abs(z) > adaptive_thr)
        adaptive_thr_list.append(adaptive_thr)

    signals = np.array(signals, dtype=int)

    # === Debug log ===
    n_trades = signals.sum()
    print(f"[DEBUG] p={p}, q={q}, thr={thr:.4f} → n_trades={n_trades}, "
          f"mean_thr={np.mean(adaptive_thr_list):.4f}, "
          f"mean_|z|={np.mean(np.abs(residuals)):.4f}")

    # tiếp tục tính sharpe/mdd
    strat_ret = returns * signals
    sharpe = np.mean(strat_ret) / (np.std(strat_ret) + 1e-8)
    mdd = np.min(np.cumsum(strat_ret) - np.maximum.accumulate(np.cumsum(strat_ret)))

    return sharpe, mdd

def debug_objective(obj_func, candidates):
    if hasattr(obj_func, "_last_runs"):
        obj_func._last_runs.clear()

    print("\n===== DEBUG RUN START =====")
    for i, x in enumerate(candidates):
        F = obj_func(x)
        print(f"- Cand#{i}: p={int(x[0])}, q={int(x[1])}, thr={float(x[2]):.4f} -> "
              f"Obj=(Sharpe_penalized={-F[0]:.6f}, MDD_penalized={F[1]:.6f})")

    runs = getattr(obj_func, "_last_runs", [])
    if len(runs) >= 2:
        print("\n--- Pairwise z-path correlation & sign equality ---")
        for i in range(len(runs)):
            for j in range(i+1, len(runs)):
                zi = runs[i]["z_path"]
                zj = runs[j]["z_path"]
                if len(zi) > 3 and len(zj) > 3 and len(zi) == len(zj):
                    corr = float(np.corrcoef(zi, zj)[0, 1])
                    sign_eq = float(np.mean(np.sign(zi) == np.sign(zj)) * 100.0)
                else:
                    corr, sign_eq = np.nan, np.nan
                print(f"  Cand#{i} vs Cand#{j}: corr={corr:.4f}, sign_eq={sign_eq:.2f}%")

        print("\n--- Raw metrics & penalty (no clip) ---")
        for idx, r in enumerate(runs):
            print(f"  Cand#{idx}: p={r['p']}, q={r['q']}, thr={r['thr']:.4f} | "
                  f"raw_Sharpe={r['raw_sharpe']:.6f}, raw_MDD={r['raw_mdd']:.6f}, "
                  f"penalty={r['penalty']:.6f}, trades={r['total_trades']}")
    print("===== DEBUG RUN END =====\n")

# -------- pymoo wrappers --------
class Tier1ARIMAProblem(ElementwiseProblem):
    def __init__(self, train, val, normalize: bool = False):
        super().__init__(n_var=2, n_obj=1, xl=np.array([1, 1]), xu=np.array([7, 7]))
        self.train = np.asarray(train, dtype=float)
        self.val = np.asarray(val, dtype=float)
        self.normalize = normalize
    def _evaluate(self, x, out, *_, **__):
        out['F'] = [arima_rmse(x, self.train, self.val, normalize=self.normalize)]

class Tier2MOGAProblem(ElementwiseProblem):
    def __init__(self, obj_func):
        super().__init__(n_var=3, n_obj=2,
                 xl=np.array([1, 1, 0.005]),
                 xu=np.array([7, 7, 0.3]))
        self.obj = obj_func
    def _evaluate(self, x, out, *_, **__):
        out['F'] = self.obj(x)

class FromTier1SeedSampling(Sampling):
    def __init__(self, seeds, n_var):
        super().__init__()
        self.seeds = np.array(seeds, dtype=float)
        self.n_var = int(n_var)
    def _do(self, problem, n_samples, **kwargs):
        n0 = min(len(self.seeds), n_samples)
        pop = self.seeds[:n0].copy()
        if n_samples > n0:
            rand = np.random.uniform(problem.xl, problem.xu, size=(n_samples - n0, self.n_var))
            pop = np.vstack([pop, rand])
        if pop.shape[1] >= 2:
            pop[:, 0] = np.clip(np.rint(pop[:, 0]), 1, 7)
            pop[:, 1] = np.clip(np.rint(pop[:, 1]), 1, 7)
        return pop