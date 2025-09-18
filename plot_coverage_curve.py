import os, json, argparse
import matplotlib.pyplot as plt

def load_result(path):
    with open(path, "r") as f:
        return json.load(f)

def plot_curve(coverage, gain=None, title="", out_png=None, out_pdf=None, show_gain=False):
    # Matplotlib styles
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300

    x = list(range(1, len(coverage) + 1))
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))

    # Coverage (primary axis)
    ln1 = ax1.plot(x, coverage, marker="o", linewidth=1.6, markersize=4.5, label="Coverage (mean)",
                   color="#1f77b4")
    ax1.set_xlabel("Number of selected folds", fontsize=11)
    ax1.set_ylabel("Coverage (mean similarity to selected)", fontsize=11)
    ax1.set_ylim(0, 1.02)
    ax1.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.6)

    ln2 = []
    if show_gain and gain is not None and len(gain) == len(coverage):
        ax2 = ax1.twinx()
        ln2 = ax2.plot(x, gain, marker="s", linewidth=1.4, markersize=3.8, label="Marginal gain",
                       color="#ff7f0e")
        ax2.set_ylabel("Marginal coverage gain", fontsize=11)
        ax2.grid(False)

    # Title + legend
    if title:
        plt.title(title, fontsize=12, pad=10)

    lines = ln1 + ln2
    labels = [l.get_label() for l in lines]
    if lines:
        plt.legend(lines, labels, frameon=False, fontsize=10, loc="lower right")

    plt.tight_layout()
    if out_png:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, bbox_inches="tight")
    if out_pdf:
        os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
        plt.savefig(out_pdf, bbox_inches="tight")
    plt.show()

def main():
    ap = argparse.ArgumentParser(description="Plot coverage gain curve from selection result JSON.")
    ap.add_argument("--result-json", required=True, help="Path to selected_{model}.json from select_representatives.py")
    ap.add_argument("--title", default="", help="Figure title")
    ap.add_argument("--out-png", default="", help="Optional output PNG path")
    ap.add_argument("--out-pdf", default="", help="Optional output PDF path")
    ap.add_argument("--show-gain", action="store_true", help="Also plot marginal gain on a twin axis")
    args = ap.parse_args()

    res = load_result(args.result_json)

    # Handle both top-level keys and nested structures
    curves = res.get("curves", {})
    coverage = curves.get("coverage", []) or res.get("coverage_curve", [])
    gain     = curves.get("gain", [])     or res.get("gain_curve", [])

    if not coverage:
        raise SystemExit("No coverage curve found in the JSON (expected 'curves.coverage' or 'coverage_curve').")

    # Simple auto-title if not provided
    mode = res.get("mode", "")
    k    = res.get("k", None)
    if not args.title:
        title = "Coverage gain curve"
        if mode:
            title += f" ({mode})"
        if k is not None:
            title += f" — final k={k}"
    else:
        title = args.title

    plot_curve(coverage, gain=gain, title=title,
               out_png=(args.out_png or None),
               out_pdf=(args.out_pdf or None),
               show_gain=args.show_gain)

if __name__ == "__main__":
    main()