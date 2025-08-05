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
DEFAULT_T2_NGEN_ARIMA = 40

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

def create_periodic_arima_objective(train, val, retrain_interval, cost):
    def objective(x):
        p, q, threshold = int(x[0]), int(x[1]), float(x[2])
        if not (0.5 < threshold < 2.5):
            return 1e3, 1e3

        history = train.copy()
        all_simple_returns = []
        all_log_returns = []
        n = len(val)

        for start in range(0, n, retrain_interval):
            end = min(start + retrain_interval, n)
            try:
                model = ARIMA(history, order=(p, 0, q)).fit()
            except Exception:
                return 1e3, 1e3

            h = end - start
            forecast = model.forecast(steps=h)
            residual_std = np.std(model.resid)

            block_volatility = np.std(val[start:end])
            if block_volatility < 0.002:  
                return 1e3, 1e3
            
            z_score = (forecast - forecast.mean()) / (residual_std + 1e-8)
            if z_score > threshold:
                signals = 1
            elif z_score < -threshold:
                signals = -1 
            else:
                signals = 0
            
            val_block_log = val[start:end]
            val_block_simple = np.exp(val_block_log) - 1

            block_simple_return = val_block_simple * signals - cost * (signals != 0)
            all_simple_returns.append(block_simple_return)

            block_simple_return = np.clip(block_simple_return, -0.9999, None)
            block_log_return = np.log(1 + block_simple_return)  
            all_log_returns.append(block_log_return)

            history = np.concatenate([history, val[start:end]])

        net_simple_returns = np.concatenate(all_simple_returns)
        net_log_returns = np.concatenate(all_log_returns)

        return -sharpe_ratio(net_simple_returns), max_drawdown(net_log_returns)

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
                         xl=np.array([1, 1, 0.5]), xu=np.array([7, 7, 2.5]))
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