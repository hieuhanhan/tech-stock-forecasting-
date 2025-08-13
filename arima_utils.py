import warnings
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
import logging

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
DEFAULT_MIN_BLOCK_VOL = 0.002

# Evaluation Utilities
def sharpe_ratio(simple_returns):
    """Annualized Sharpe on daily simple returns."""
    std = float(np.std(simple_returns))
    return 0.0 if std == 0.0 else (float(np.mean(simple_returns)) / std) * np.sqrt(252.0)

def max_drawdown(log_returns):
    """Max drawdown computed on cumulative log-returns."""
    if len(log_returns) == 0:
        return 0.0
    cum = np.exp(np.cumsum(log_returns))
    peak = np.maximum.accumulate(cum)
    # avoid /0 with eps
    return float(np.max((peak - cum) / (peak + np.finfo(float).eps)))

# Tier-1: RMSE Objective
def arima_rmse(params, train, val, normalize: bool = False) -> float:
    """
    Fit ARIMA(p,0,q) on train; forecast len(val); return RMSE or NRMSE.
    - p,q are clipped to [0,7]
    - normalize=True returns RMSE / std(val) (NRMSE)
    """
    p, q = map(int, params)
    p = int(np.clip(p, 1, 7))
    q = int(np.clip(q, 1, 7))

    # Fit & forecast (no caching across data objects; keeping it simple/reliable)
    try:
        model = ARIMA(train, order=(p, 0, q)).fit()
        forecast = model.forecast(steps=len(val))
        rmse = sqrt(mean_squared_error(val, forecast))
        if normalize:
            denom = float(np.std(val)) or 1.0
            return float(rmse / denom)
        return float(rmse)
    except Exception:
        # on failure, return large penalty
        return 1e3

# Tier-2: Periodic ARIMA Strategy Objective

def create_periodic_arima_objective(
    train: np.ndarray,
    val: np.ndarray,
    retrain_interval: int,
    cost_per_trade: float,
    min_block_vol: float = DEFAULT_MIN_BLOCK_VOL,
):
    """
    Build a 2-objective function: minimize (-Sharpe(simple)), minimize (MaxDrawdown(log)).
    Strategy:
      - For each block of length retrain_interval on val:
        * Fit ARIMA(p,0,q) on growing 'history'
        * Forecast h steps; compute z = (forecast - mean)/resid_std (vector)
        * Position per step: pos = sign(z) with threshold
        * Cost charged per CHANGE of position (trades), not per day-in-position
    x = [p, q, threshold], with p,q ∈ [0,7], threshold ∈ (0.5, 2.5)
    """
    train = np.asarray(train, dtype=float)
    val = np.asarray(val, dtype=float)

    def objective(x):
        # sanitize vars
        p = int(np.clip(np.rint(x[0]), 1, 7))
        q = int(np.clip(np.rint(x[1]), 1, 7))
        threshold = float(x[2])

        history = train.copy()
        simple_all = []
        log_all = []
        n = len(val)

        for start in range(0, n, retrain_interval):
            end = min(start + retrain_interval, n)
            block_log = val[start:end]           # log-returns
            h = end - start
            if h <= 0:
                continue

            try:
                model = ARIMA(history, order=(p, 0, q)).fit()
            except Exception:
                return 1e3, 1e3

            forecast = np.asarray(model.forecast(steps=h), dtype=float)
            resid_std = float(np.std(model.resid)) + 1e-8

            # adaptive threshold limit
            pred_range = np.nanmax(np.abs(forecast - forecast.mean()))
            upper_thr = max(0.01, min(pred_range, 2.0))
            if threshold > upper_thr:
                return 1e3, 1e3

            if float(np.std(block_log)) < float(min_block_vol):
                return 1e3, 1e3

            z = (forecast - forecast.mean()) / resid_std
            pos = np.where(z > threshold, 1, np.where(z < -threshold, -1, 0))

            buy_count = np.sum(pos == 1)
            sell_count = np.sum(pos == -1)
            if (buy_count + sell_count) == 0:
                logging.warning(f"NO TRADE in block {start}-{end} for params p={p},q={q},thr={threshold}")
                return -np.inf, 0.0

            # Convert log->simple, apply position & trading cost per change
            block_simple = np.exp(block_log) - 1.0
            trades = np.abs(np.diff(np.r_[0, pos]))
            cost_vec = cost_per_trade * trades
            ret_simple = block_simple * np.roll(pos, 1) - cost_vec  # shift for next day entry
            ret_simple[0] = 0.0

            ret_simple = np.clip(ret_simple, -0.9999, None)
            ret_log = np.log1p(ret_simple)

            simple_all.append(ret_simple)
            log_all.append(ret_log)
            history = np.concatenate([history, block_log])

        if not simple_all:
            return 1e3, 1e3

        net_simple = np.concatenate(simple_all)
        net_log = np.concatenate(log_all)
        return -sharpe_ratio(net_simple), max_drawdown(net_log)

    return objective

# pymoo Problem definitions
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
                         xl=np.array([1, 1, 0.01]), xu=np.array([7, 7, 2.0]))
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