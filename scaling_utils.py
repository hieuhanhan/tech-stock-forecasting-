from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

@dataclass
class ScalerSpec:
    mode: str  # "standard" | "minmax" | "hybrid"
    minmax_cols: Optional[List[str]] = None  # only used when mode == "hybrid"

class HybridScaler:
    """Scale some columns by MinMax and the rest by StandardScaler.

    Notes
    -----
    - This class is intentionally lightweight and deterministic to be pickle‑safe.
    - Expect DataFrame inputs for fit/transform to preserve column names & order.
    """

    def __init__(self, minmax_cols: List[str], all_feature_cols: List[str]):
        self.minmax_cols = list(minmax_cols)
        self.std_cols = [c for c in all_feature_cols if c not in self.minmax_cols]
        self.mm = MinMaxScaler()
        self.ss = StandardScaler()

    # --- sklearn‑like API ---
    def fit(self, Xdf: pd.DataFrame):
        if self.minmax_cols:
            self.mm.fit(Xdf[self.minmax_cols].to_numpy(dtype=np.float32))
        if self.std_cols:
            self.ss.fit(Xdf[self.std_cols].to_numpy(dtype=np.float32))
        return self

    def transform(self, Xdf: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=Xdf.index)
        if self.minmax_cols:
            out[self.minmax_cols] = self.mm.transform(Xdf[self.minmax_cols].to_numpy(dtype=np.float32))
        if self.std_cols:
            out[self.std_cols] = self.ss.transform(Xdf[self.std_cols].to_numpy(dtype=np.float32))
        # Return in a stable order: minmax first then std, or just std if no minmax
        return out[self.minmax_cols + self.std_cols] if self.minmax_cols else out[self.std_cols]

    # Optional metadata for logs
    def meta(self) -> Dict[str, Any]:
        return {"type": "hybrid", "minmax_cols": self.minmax_cols, "std_cols": self.std_cols}


# --------- Builder ---------

def build_scaler(spec: ScalerSpec, feature_cols: List[str]) -> Tuple[object, Dict[str, Any]]:
    """Factory to build a scaler and its serializable metadata.

    Returns
    -------
    scaler : StandardScaler | MinMaxScaler | HybridScaler
    meta   : Dict with type and relevant fields for audit
    """
    if spec.mode == "standard":
        return StandardScaler(), {"type": "standard"}
    if spec.mode == "minmax":
        return MinMaxScaler(), {"type": "minmax"}
    if spec.mode == "hybrid":
        if not spec.minmax_cols:
            raise ValueError("[ERROR] minmax_cols must be provided for hybrid mode.")
        not_in = [c for c in spec.minmax_cols if c not in feature_cols]
        if not_in:
            raise ValueError(f"[ERROR] minmax_cols not in feature columns: {not_in}")
        hs = HybridScaler(spec.minmax_cols, feature_cols)
        return hs, hs.meta()
    raise ValueError(f"Unknown scaler mode: {spec.mode}")


# --------- Backward‑compat shim for old pickles ---------

def install_unpickle_shim():
    """Install a temporary attribute on the current __main__ module so that
    pickles saved when HybridScaler lived in __main__ can be unpickled.

    Call this before joblib.load(...) in scripts that read older artifacts.
    """
    import sys
    mod_main = sys.modules.get("__main__")
    if mod_main is not None and not hasattr(mod_main, "HybridScaler"):
        setattr(mod_main, "HybridScaler", HybridScaler)