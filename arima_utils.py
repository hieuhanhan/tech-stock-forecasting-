import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling

LOG_DIR = "logs"
GA_POP_SIZE = 35
GA_N_GEN = 25
BO_N_CALLS = 50
TOP_N_SEEDS = 5
BASE_COST = 0.0005
SLIPPAGE = 0.0002
DEFAULT_T2_NGEN_ARIMA = 30

# Evaluation Metrics

def arima_rmse(params, train, val):
    p, q = map(int, params)
    try:
        model = ARIMA(train, order=(p, 0, q)).fit()
        forecast = model.forecast(steps=len(val))
        return sqrt(mean_squared_error(val, forecast))
    except Exception:
        return 1e3

def sharpe_ratio(r):
    std = np.std(r)
    return 0.0 if std == 0 else (np.mean(r) / std) * np.sqrt(252)

def max_drawdown(r):
    if len(r) == 0:
        return 0.0
    cum = np.exp(np.cumsum(r))
    peak = np.maximum.accumulate(cum)
    return np.max((peak - cum) / (peak + np.finfo(float).eps))

# Objective Wrapper

def create_periodic_arima_objective(train, val, retrain_interval, cost=0.0005):
    def objective(x):
        p, q, rel_thresh = int(x[0]), int(x[1]), float(x[2])
        if not (0.0 < rel_thresh < 1.0):
            return 1e3, 1e3

        history = train.copy()
        all_returns = []
        n = len(val)

        for start in range(0, n, retrain_interval):
            end = min(start + retrain_interval, n)
            try:
                model = ARIMA(history, order=(p, 0, q)).fit()
            except Exception:
                return 1e3, 1e3

            h = end - start
            forecast = model.forecast(steps=h)
            threshold_value = np.quantile(forecast, rel_thresh)
            signals = (forecast > threshold_value).astype(int)
            if signals.sum() == 0:
                return 1e3, 1e3

            block_returns = val[start:end] * signals - cost * signals
            all_returns.append(block_returns)
            history = np.concatenate([history, val[start:end]])

        net_returns = np.concatenate(all_returns)
        return -sharpe_ratio(net_returns), max_drawdown(net_returns)

    return objective

# Pymoo Definitions

class Tier1ARIMAProblem(ElementwiseProblem):
    def __init__(self, train, val):
        super().__init__(n_var=2, n_obj=1, xl=np.array([1, 1]), xu=np.array([7, 7]))
        self.train = train
        self.val = val

    def _evaluate(self, x, out, *_, **__):
        out['F'] = [arima_rmse(x, self.train, self.val)]

class Tier2MOGAProblem(ElementwiseProblem):
    def __init__(self, obj_func):
        super().__init__(n_var=3, n_obj=2,
                         xl=np.array([1, 1, 0.0]), xu=np.array([7, 7, 1.0]))
        self.obj = obj_func

    def _evaluate(self, x, out, *_, **__):
        out['F'] = self.obj(x)

class FromTier1SeedSampling(Sampling):
    def __init__(self, seeds, n_var):
        super().__init__()
        self.seeds = np.array(seeds)
        self.n_var = n_var

    def _do(self, problem, n_samples, **kwargs):
        n0 = min(len(self.seeds), n_samples)
        pop = self.seeds[:n0].copy()
        if n_samples > n0:
            rand = np.random.uniform(problem.xl, problem.xu,
                                     size=(n_samples - n0, self.n_var))
            pop = np.vstack([pop, rand])
        return pop