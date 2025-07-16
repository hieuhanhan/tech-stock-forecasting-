import pandas as pd
import json
import os
import numpy as np
from tqdm import tqdm

# --- Import các thư viện cần thiết ---
from statsmodels.tsa.arima.model import ARIMA
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
# --- THÊM VÀO: Thư viện để chạy song song ---
import multiprocessing as mp

# ===================================================================
# 1. CÁC HÀM HỖ TRỢ (Không thay đổi)
# ===================================================================
def calculate_sharpe_ratio(daily_returns):
    if np.std(daily_returns) == 0: return 0.0
    return (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252)

def calculate_max_drawdown(daily_returns):
    if len(daily_returns) == 0: return 0.0
    cumulative_returns = np.exp(np.cumsum(daily_returns))
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (running_max - cumulative_returns) / running_max
    return np.max(drawdown)

def create_final_moga_objective_arima(train_log_returns, actual_log_returns, transaction_cost=0.0005):
    def moga_objective_function(params):
        p, q, action_threshold = int(params[0]), int(params[1]), params[2]
        try:
            model = ARIMA(train_log_returns, order=(p, 0, q))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=len(actual_log_returns)).values
            signals = (forecast > action_threshold).astype(int)
            net_returns = (actual_log_returns * signals) - (signals * transaction_cost)
            sharpe = calculate_sharpe_ratio(net_returns)
            drawdown = calculate_max_drawdown(net_returns)
            return (-sharpe, drawdown)
        except Exception:
            return (1e9, 1e9)
    return moga_objective_function

class AdvancedARIMAProblem(ElementwiseProblem):
    def __init__(self, objective_func):
        super().__init__(n_var=3, n_obj=2, xl=np.array([1, 1, 0.0]), xu=np.array([7, 7, 0.01]))
        self.objective_func = objective_func
    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.objective_func(x)

# ===================================================================
# 2. HÀM TỐI ƯU HÓA CHO MỘT FOLD DUY NHẤT (Đã được sửa đổi)
# ===================================================================
def optimize_single_fold(fold_id, folds_summary_map, folds_dir):
    """
    Hàm này chứa toàn bộ logic để xử lý một fold duy nhất.
    Nó sẽ được gọi song song bởi nhiều tiến trình.
    """
    fold_info = folds_summary_map.get(fold_id)
    if not fold_info:
        return None

    train_path = os.path.join(folds_dir, fold_info['train_path_arima_prophet'])
    val_path = os.path.join(folds_dir, fold_info['val_path_arima_prophet'])
    
    try:
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
    except FileNotFoundError:
        return None

    train_log_returns = train_df['Log_Returns'].dropna()
    actual_returns = val_df['Log_Returns'].values

    moga_objective = create_final_moga_objective_arima(train_log_returns, actual_returns)
    problem = AdvancedARIMAProblem(moga_objective)
    algorithm = NSGA2(pop_size=50)
    
    res = minimize(problem, algorithm, ('n_gen', 40), seed=42, verbose=False)

    return {
        'fold_id': fold_id,
        'ticker': fold_info['ticker'],
        'pareto_front': [
            {'p': int(sol[0]), 'q': int(sol[1]), 'threshold': sol[2], 'sharpe_ratio': -obj[0], 'max_drawdown': obj[1]}
            for sol, obj in zip(res.X, res.F)
        ]
    }

# ===================================================================
# 3. SCRIPT CHÍNH (Đã được sửa đổi để chạy song song)
# ===================================================================
if __name__ == "__main__":
    print("Starting FINAL Multi-Objective Optimization (MOGA) for ARIMA in PARALLEL...")

    # Tải các file cần thiết
    results_dir = 'data/tuning_results'
    folds_dir = 'data/processed_folds'
    folds_summary_path = os.path.join(folds_dir, 'folds_summary.json')
    representative_folds_path = os.path.join(folds_dir, 'shared_meta', 'representative_fold_ids.json')
    
    with open(folds_summary_path, 'r') as f:
        all_folds_summary = json.load(f)
    with open(representative_folds_path, 'r') as f:
        representative_fold_ids = json.load(f)
    folds_summary_map = {item['global_fold_id']: item for item in all_folds_summary}

    # --- THAY ĐỔI LỚN: SỬ DỤNG MULTIPROCESSING ---
    # Lấy số lượng CPU có sẵn trên máy ảo
    num_processes = mp.cpu_count()
    print(f"Found {num_processes} CPU cores. Starting parallel processing...")

    # Tạo một "pool" các tiến trình làm việc
    with mp.Pool(processes=num_processes) as pool:
        # Chuẩn bị các tham số cho mỗi tác vụ
        tasks = [(fold_id, folds_summary_map, folds_dir) for fold_id in representative_fold_ids]
        
        # Phân phối các tác vụ cho các tiến trình và hiển thị thanh tiến trình
        results = list(tqdm(pool.starmap(optimize_single_fold, tasks), total=len(tasks)))

    # Lọc bỏ các kết quả None (nếu có lỗi xảy ra)
    final_moga_results = [res for res in results if res is not None]

    # --- Lưu kết quả cuối cùng ---
    print("\n--- FINAL MOGA Tuning Complete for ARIMA! ---")
    moga_results_path = os.path.join(results_dir, 'final_moga_arima_results.json')
    with open(moga_results_path, 'w') as f:
        json.dump(final_moga_results, f, indent=4)
    print(f"Final ARIMA MOGA results saved to: {moga_results_path}")
