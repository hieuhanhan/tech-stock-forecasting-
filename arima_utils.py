import warnings
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from arch import arch_model
import logging

# === Constants ===
LOG_DIR = "logs"
BASE_COST = 0.0005
SLIPPAGE = 0.0002
DEFAULT_T2_NGEN_ARIMA = 40
DEFAULT_MIN_BLOCK_VOL = 0.0015

# -------- Metrics --------
def suppress_warnings():
    # sklearn convergence
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # statsmodels / ARIMA warnings
    warnings.filterwarnings("ignore", message="Non-invertible")
    warnings.filterwarnings("ignore", message="Maximum Likelihood optimization failed")
    warnings.filterwarnings("ignore", message="Non-stationary starting autoregressive parameters")
    warnings.filterwarnings("ignore", message="Too few observations")
    warnings.filterwarnings("ignore", category=FutureWarning)

def sharpe_ratio(simple_returns: np.ndarray) -> float:
    r = np.asarray(simple_returns, dtype=float)
    if r.size < 2:
        return 0.0
    std = float(np.std(r, ddof=1))
    return 0.0 if std == 0.0 else (float(np.mean(r)) / std) * np.sqrt(252.0)

def max_drawdown(data: np.ndarray, input_type: str = "simple") -> float:
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

# -------- Tier-2: Continuous Objective --------
def create_continuous_arima_objective(
    train: np.ndarray,
    val: np.ndarray,
    retrain_interval: int,
    cost_per_turnover: float,
    min_block_vol: float = DEFAULT_MIN_BLOCK_VOL,
    scale_factor: float = 1000.0,
    arch_rescale_flag: bool = False,
    debug: bool = True,
    thr_min: float = 0.01,
    thr_max: float = 0.6,
    pos_clip: float = 1.0,
):
    """
    Continuous strategy:
      z_t = mean_t / sqrt(var_t)
      pos_t = clip(z_t / thr, -pos_clip, pos_clip)
      turnover_t = |pos_t - pos_{t-1}|
      cost_t = (BASE_COST + SLIPPAGE) * turnover_t
      ret_t = (exp(r_t) - 1) * pos_t - cost_t
    """
    train = np.asarray(train, dtype=float)
    val = np.asarray(val, dtype=float)

    # penalties
    W_THR_OUT = 10.0
    W_FIT_FAIL = 5.0
    W_LOW_VOL = 2.0
    W_NO_ACT = 0.5
    EPS = 1e-8

    def _skew(x: np.ndarray, eps=1e-12):
        x = np.asarray(x, dtype=float)
        if x.size < 3: return 0.0
        m = float(np.mean(x)); s = float(np.std(x, ddof=1))
        if s < eps: return 0.0
        m3 = float(np.mean((x - m) ** 3))
        return float(m3 / (s**3 + eps))

    stats_map = {}

    def objective(x):
        nonlocal stats_map
        p = int(np.clip(np.rint(x[0]), 1, 7))
        q = int(np.clip(np.rint(x[1]), 1, 7))
        thr = float(x[2])

        penalty = 0.0
        if thr < thr_min: penalty += W_THR_OUT * (thr_min - thr); thr = thr_min
        elif thr > thr_max: penalty += W_THR_OUT * (thr - thr_max); thr = thr_max

        history = train.copy()
        simple_all, log_all = [], []
        total_turnover = 0.0
        prev_pos_last = 0.0

        total_blocks = 0
        n = len(val)

        for start in range(0, n, retrain_interval):
            end = min(start + retrain_interval, n)
            block_log = val[start:end]
            h = end - start
            if h <= 0:
                continue
            total_blocks += 1

            block_vol = float(np.std(block_log))
            if block_vol < float(min_block_vol):
                penalty += W_LOW_VOL * (min_block_vol - block_vol)
                history = np.concatenate([history, block_log])
                if debug:
                    logging.debug(f"[BLK {start:04d}-{end:04d}] SKIP vol={block_vol:.3e} < {min_block_vol:.3e}")
                continue

            # Fit ARIMA + GARCH
            try:
                arima_fit = ARIMA(history, order=(p, 0, q)).fit()
                resid = np.asarray(arima_fit.resid, dtype=float).ravel()

                garch = arch_model(resid * scale_factor, mean="Zero", vol="Garch",
                                   p=1, q=1, dist="normal", rescale=arch_rescale_flag)
                g_res = garch.fit(disp='off')

                f_var = np.asarray(g_res.forecast(horizon=h).variance.iloc[-1]).ravel()
                f_mu  = np.asarray(arima_fit.forecast(steps=h)).ravel()
                f_sig = np.sqrt(np.maximum(f_var / (scale_factor**2 + EPS), EPS))

                if debug:
                    r_mean = float(np.mean(resid))
                    r_std  = float(np.std(resid, ddof=1))
                    r_skew = _skew(resid)
            except Exception as e:
                logging.debug(f"[FIT_FAIL] p={p}, q={q}, err={e}")
                penalty += W_FIT_FAIL
                history = np.concatenate([history, block_log])
                continue

            # z-score & continuous position
            z = f_mu / (f_sig + EPS)

            # Continuous sizing
            pos_now = np.clip(z / (thr + EPS), -pos_clip, pos_clip)

            # Transaction cost by turnover
            signals_full = np.concatenate([[prev_pos_last], pos_now])
            turnover = np.abs(np.diff(signals_full))
            cost_vec = cost_per_turnover * turnover
            total_turnover += float(turnover.sum())

            # PnL
            block_simple = np.exp(block_log) - 1.0
            ret_simple = block_simple * pos_now - cost_vec
            ret_simple = np.clip(ret_simple, -0.9999, None)
            ret_log = np.log1p(ret_simple)

            simple_all.append(ret_simple)
            log_all.append(ret_log)

            # debug block
            if debug:
                zmin, zmax, zmean = float(np.min(z)), float(np.max(z)), float(np.mean(z))
                pmin, pmax, pmean = float(np.min(pos_now)), float(np.max(pos_now)), float(np.mean(pos_now))
                logging.debug(
                    f"[BLK {start:04d}-{end:04d}] mode=continuous | p={p},q={q},thr={thr:.3f} | "
                    f"vol={block_vol:.3e}, turnover={turnover.sum():.3f} | "
                    f"z[{zmin:.2f}/{zmean:.2f}/{zmax:.2f}] pos[{pmin:.2f}/{pmean:.2f}/{pmax:.2f}] | "
                    f"resid[{r_mean:.2e},{r_std:.2e},{r_skew:.2f}]"
                )

            history = np.concatenate([history, block_log])
            prev_pos_last = float(pos_now[-1])

        # guard if no valid blocks
        if not simple_all:
            penalty += W_FIT_FAIL
            if debug:
                logging.debug(f"[KPI] blocks={total_blocks}, turnover={total_turnover:.3f} (NO_VALID_BLOCKS)")
            F0 = 1.0 + penalty; F1 = 1.0 + penalty
            stats_map[(p, q, round(thr, 6))] = dict(turnover=total_turnover, raw_sharpe=0.0, raw_mdd=0.0, penalty=penalty)
            return F0, F1

        net_simple = np.concatenate(simple_all)
        net_log    = np.concatenate(log_all)

        if total_turnover == 0.0:
            penalty += W_NO_ACT

        raw_sharpe = sharpe_ratio(net_simple)
        raw_mdd    = max_drawdown(net_log, input_type="log")

        f0, f1 = -raw_sharpe, raw_mdd
        if not np.isfinite(f0): penalty += W_FIT_FAIL; f0 = 1.0
        if not np.isfinite(f1): penalty += W_FIT_FAIL; f1 = 1.0

        if debug:
            logging.debug(
                f"[KPI] blocks={total_blocks}, turnover={total_turnover:.3f}, "
                f"Sharpe={raw_sharpe:.3f}, MDD={raw_mdd:.3f}, Penalty={penalty:.3f}"
            )

        # save meta for callback
        stats_map[(p, q, round(thr, 6))] = dict(
            turnover=float(total_turnover),
            raw_sharpe=float(raw_sharpe),
            raw_mdd=float(raw_mdd),
            penalty=float(penalty),
        )

        return float(f0 + penalty), float(f1 + penalty)

    # expose for callback
    objective.stats_map = stats_map
    return objective

# -------- pymoo wrappers (continuous) --------
class Tier1ARIMAProblem(ElementwiseProblem):
    def __init__(self, train, val, normalize: bool = False):
        super().__init__(n_var=2, n_obj=1, xl=np.array([1, 1]), xu=np.array([7, 7]))
        self.train = np.asarray(train, dtype=float)
        self.val = np.asarray(val, dtype=float)
        self.normalize = normalize
    def _evaluate(self, x, out, *_, **__):
        out['F'] = [arima_rmse(x, self.train, self.val, normalize=self.normalize)]

class Tier2MOGAProblem(ElementwiseProblem):
    def __init__(self, obj_func, thr_bounds=(0.01, 0.6)):
        super().__init__(n_var=3, n_obj=2,
                         xl=np.array([1, 1, thr_bounds[0]]),
                         xu=np.array([7, 7, thr_bounds[1]]))
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
    
# -------- Pareto helpers --------
def is_nondominated(F: np.ndarray) -> np.ndarray:
    n = F.shape[0]
    nd = np.ones(n, dtype=bool)
    for i in range(n):
        if not nd[i]: 
            continue
        fi = F[i]
        # j dominates i if fj <= fi all & < in at least one
        dom = np.all(F <= fi + 1e-12, axis=1) & np.any(F < fi - 1e-12, axis=1)
        dom[i] = False
        if np.any(dom):
            nd[i] = False
    return nd

def knee_index(F: np.ndarray) -> int:
    # F already minimized objectives (f0=-Sharpe, f1=MDD)
    f = F.copy()
    f = (f - f.min(axis=0)) / (np.ptp(f, axis=0) + 1e-12)
    d = np.sqrt((f**2).sum(axis=1))
    return int(np.argmin(d))