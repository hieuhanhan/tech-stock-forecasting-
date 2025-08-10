from dataclasses import dataclass
from typing import List, Dict

@dataclass
class ModelConfig:
    meta_mode: str  # 'arima' or 'lstm'
    k_min: int
    k_max: int
    n_per_cluster: int
    n_per_ticker: int
    desired_tickers: List[str]
    min_per_cluster: int
    distance_quantile: float  # e.g., 0.3 for 30th percentile
    force_quantile: float     # e.g., 0.6 for 60th percentile


DEFAULT_CONFIGS: Dict[str, ModelConfig] = {
    'arima': ModelConfig(
        meta_mode='arima',
        k_min=5, k_max=25,
        n_per_cluster=1,
        n_per_ticker=2,
        desired_tickers=["AAPL","AMZN","GOOGL","META","MSFT","NVDA","TSLA"],
        min_per_cluster=1,
        distance_quantile=0.30,
        force_quantile=0.60,
    ),
    'lstm': ModelConfig(
        meta_mode='lstm',
        k_min=5, k_max=25,
        n_per_cluster=1,
        n_per_ticker=2,
        desired_tickers=["AAPL","AMZN","GOOGL","META","MSFT","NVDA","TSLA"],
        min_per_cluster=1,
        distance_quantile=0.30,
        force_quantile=0.60,
    )
}