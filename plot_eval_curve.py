#!/usr/bin/env python3
import os
import re
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

LOG_PATTERN = re.compile(
    r"k=\s*(\d+)\s*\|\s*cov_mean=([\d\.]+).*?diversity_mean=([\d\.]+).*?selected=(\d+)"
)

def parse_eval_log(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            m = LOG_PATTERN.search(ln)
            if m:
                k = int(m.group(1))
                cov = float(m.group(2))
                div = float(m.group(3))
                sel = int(m.group(4))
                rows.append((k, cov, div, sel))
    rows.sort(key=lambda x: x[0])
    return rows

def find_elbow(K, C, eps=0.005):
    gains = np.r_[C[0], np.diff(C)]
    elbow_idx = np.argmax(gains < eps) if (gains < eps).any() else len(K) - 1
    return int(K[elbow_idx]), float(C[elbow_idx])

def main():
    ap = argparse.ArgumentParser(description="Plot Coverage/Diversity vs k from --eval_only log output.")
    ap.add_argument("--model_type", required=True, choices=["arima", "lstm"], help="Model label for titles/files.")
    ap.add_argument("--log_path", required=True, help="Path to eval_k_log.txt from --eval_only output.")
    ap.add_argument("--outdir", default="data/figures", help="Directory to save figures.")
    ap.add_argument("--eps", type=float, default=0.005, help="Elbow threshold on coverage gain (default 0.005).")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    rows = parse_eval_log(args.log_path)
    if not rows:
        sys.exit("No eval rows parsed. Ensure the log file comes from --eval_only output.")

    K = np.array([r[0] for r in rows], dtype=int)
    C = np.array([r[1] for r in rows], dtype=float)  # coverage
    D = np.array([r[2] for r in rows], dtype=float)  # diversity

    k_star, c_star = find_elbow(K, C, eps=args.eps)

    # Coverage vs k
    plt.figure(figsize=(7, 5))
    plt.plot(K, C, marker="o")
    plt.axvline(k_star, linestyle="--")
    plt.scatter([k_star], [c_star], zorder=5)
    plt.title(f"{args.model_type.upper()} — Coverage vs k (representative folds)")
    plt.xlabel("k (number of representative folds)")
    plt.ylabel("Coverage mean")
    plt.text(k_star, c_star, f"  elbow≈{k_star}\n  cov={c_star:.3f}", va="bottom")
    plt.grid(True)
    plt.tight_layout()
    cov_path = os.path.join(args.outdir, f"{args.model_type}_coverage_vs_k.png")
    plt.savefig(cov_path, dpi=300)

    # Diversity vs k
    plt.figure(figsize=(7, 5))
    plt.plot(K, D, marker="s")
    plt.title(f"{args.model_type.upper()} — Diversity vs k")
    plt.xlabel("k")
    plt.ylabel("Dissimilarity mean (higher = more diverse)")
    plt.grid(True)
    plt.tight_layout()
    div_path = os.path.join(args.outdir, f"{args.model_type}_diversity_vs_k.png")
    plt.savefig(div_path, dpi=300)

    print(f"Saved:\n  {cov_path}\n  {div_path}\nElbow ≈ k={k_star}, cov={c_star:.3f}")

if __name__ == "__main__":
    main()