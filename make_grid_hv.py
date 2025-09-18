from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import re, math, argparse, sys

# ---------------- helpers ----------------
def paste_centered(canvas: Image.Image, img_path: Path, box):
    x, y, w, h = box
    try:
        im = Image.open(img_path).convert("RGB")
    except Exception:
        d = ImageDraw.Draw(canvas)
        d.rectangle([x, y, x+w, y+h], outline="gray", width=2)
        d.text((x+10, y+10), f"Missing: {img_path.name}", fill="red")
        return
    im.thumbnail((w, h), Image.LANCZOS)
    ox = x + (w - im.width)//2
    oy = y + (h - im.height)//2
    canvas.paste(im, (ox, oy))

def parse_fold_num(name: str):
    m = re.search(r"hv_fold_?(\d+)$", name)  
    return int(m.group(1)) if m else None

def collect_images(base_dir: Path):
    mapping = {}
    if not base_dir.exists():
        print(f"[ERROR] Not found: {base_dir}", file=sys.stderr)
        return mapping

    for fdir in sorted(base_dir.glob("hv_fold*"), key=lambda p: parse_fold_num(p.name) or 0):
        fnum = parse_fold_num(fdir.name)
        if fnum is None:
            continue
        imgs = {}
        for p in fdir.glob("*.png"):
            m = re.search(r"_int(10|20|42)\.png$", p.name)
            if m:
                imgs[m.group(1)] = p
        if imgs:
            mapping[fnum] = imgs
    print(f"[INFO] Scanned {base_dir} -> {len(mapping)} folds.")
    return mapping

# --------------- renderer ----------------
def render_grids_no_labels(model_name: str,
                           data_map: dict,
                           out_dir: Path,
                           out_prefix: str,
                           intervals=("10","20","42"),
                           cell_w=720, cell_h=440,
                           padding=28,
                           max_rows_per_image=10):
    if not data_map:
        print(f"[WARN] No data for {model_name}. Skipped.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    folds = sorted(data_map.keys())
    cols = len(intervals)
    per_image = max_rows_per_image
    num_images = math.ceil(len(folds) / per_image)

    grid_w = cols*cell_w + (cols+1)*padding

    for i in range(num_images):
        batch = folds[i*per_image : (i+1)*per_image]
        rows = len(batch)
        grid_h = rows*cell_h + (rows+1)*padding
        W, H = grid_w, grid_h

        canvas = Image.new("RGB", (W, H), "white")

        for r, fold in enumerate(batch):
            y = padding + r*(cell_h + padding)
            for c, iv in enumerate(intervals):
                x = padding + c*(cell_w + padding)
                box = (x, y, cell_w, cell_h)
                img_path = data_map.get(fold, {}).get(iv)
                if img_path:
                    paste_centered(canvas, img_path, box)
                else:
                    d = ImageDraw.Draw(canvas)
                    d.rectangle([x, y, x+cell_w, y+cell_h], outline="gray", width=2)
                    d.text((x+10, y+10), "No image", fill="gray")

        suffix = "" if num_images == 1 else f"_part{i+1}"
        out_path = out_dir / f"{out_prefix}{suffix}.png"
        canvas.save(out_path, dpi=(150,150))
        print(f"[OK] Saved: {out_path}")

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="Make HV grids (no labels) for ARIMA–GARCH & LSTM.")
    ap.add_argument("--arima_dir", default="data/figures/tier2_arima/hv_arima")
    ap.add_argument("--lstm_dir", default="data/figures/tier2_lstm/hv_lstm")
    ap.add_argument("--out_dir", default="figures/grids_hv_nolabels")
    ap.add_argument("--cell_w", type=int, default=720)
    ap.add_argument("--cell_h", type=int, default=440)
    ap.add_argument("--padding", type=int, default=28)
    ap.add_argument("--max-rows", type=int, default=9999)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    # ARIMA–GARCH
    arima_map = collect_images(Path(args.arima_dir))
    render_grids_no_labels(
        model_name="ARIMA–GARCH",
        data_map=arima_map,
        out_dir=out_dir,
        out_prefix="hv_arima_grid",
        cell_w=args.cell_w, cell_h=args.cell_h,
        padding=args.padding,
        max_rows_per_image=args.max_rows
    )

    # LSTM
    lstm_map = collect_images(Path(args.lstm_dir))
    render_grids_no_labels(
        model_name="LSTM",
        data_map=lstm_map,
        out_dir=out_dir,
        out_prefix="hv_lstm_grid",
        cell_w=args.cell_w, cell_h=args.cell_h,
        padding=args.padding,
        max_rows_per_image=args.max_rows
    )

if __name__ == "__main__":
    main()