# make_grid_equity_3x2_pdf.py
# Python 3.8+ ; Pillow required: pip install pillow

from pathlib import Path
from PIL import Image
import re, math, argparse, sys

# --------- helpers ---------
def parse_fold_interval(name: str):
    # accept: fold_113_20  -> (113, 20)
    m = re.match(r"fold_(\d+)_(\d+)$", name)
    if not m: 
        return None
    return int(m.group(1)), int(m.group(2))

def collect_equity_images(base_dir: Path):
    """
    Quét các thư mục con dạng fold_{id}_{interval} và lấy file equity_curve.png
    Trả về list[(sort_key, path)] để sắp xếp ổn định: theo fold, rồi interval (10,20,42).
    """
    if not base_dir.exists():
        print(f"[WARN] Not found: {base_dir}", file=sys.stderr)
        return []

    out = []
    for d in base_dir.iterdir():
        if not d.is_dir(): 
            continue
        key = parse_fold_interval(d.name)
        if not key:
            continue
        img = d / "equity_curve.png"
        if img.exists():
            out.append((key, img))
    # sort by fold then interval
    out.sort(key=lambda t: (t[0][0], t[0][1]))
    return [p for _, p in out]

def paste_centered(canvas: Image.Image, img_path: Path, box):
    x, y, w, h = box
    try:
        im = Image.open(img_path).convert("RGB")
    except Exception:
        # ô rỗng khi thiếu ảnh
        return
    im.thumbnail((w, h), Image.LANCZOS)
    ox = x + (w - im.width)//2
    oy = y + (h - im.height)//2
    canvas.paste(im, (ox, oy))

def build_pdf(images, out_pdf: Path, cols=2, rows=3, cell_w=720, cell_h=440, padding=28, bg="white"):
    """
    Ghép images vào nhiều trang (mỗi trang rows×cols), không chữ, lưu 1 file PDF.
    """
    if not images:
        print(f"[WARN] No images -> skip {out_pdf.name}")
        return

    per_page = rows * cols
    grid_w = cols*cell_w + (cols+1)*padding
    grid_h = rows*cell_h + (rows+1)*padding
    W, H = grid_w, grid_h

    pages = []
    for i in range(0, len(images), per_page):
        batch = images[i:i+per_page]
        page = Image.new("RGB", (W, H), bg)
        for j, img_path in enumerate(batch):
            r, c = divmod(j, cols)
            x = padding + c*(cell_w + padding)
            y = padding + r*(cell_h + padding)
            paste_centered(page, img_path, (x, y, cell_w, cell_h))
        pages.append(page)

    # lưu PDF (chỉ PDF, không xuất PNG)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(out_pdf, save_all=True, append_images=pages[1:])
    print(f"[OK] Saved PDF -> {out_pdf}")

# --------- CLI ---------
def main():
    ap = argparse.ArgumentParser(description="Make 3x2 grids (equity_curve.png) into a single PDF per model.")
    ap.add_argument("--arima_dir", default="viz_outputs/arima", type=str)
    ap.add_argument("--lstm_dir",  default="viz_outputs/lstm",  type=str)
    ap.add_argument("--out_dir",   default="figures/grids_equity", type=str)
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--cell_w", type=int, default=720)
    ap.add_argument("--cell_h", type=int, default=440)
    ap.add_argument("--pad",    type=int, default=28)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    # ARIMA–GARCH
    arima_imgs = collect_equity_images(Path(args.arima_dir))
    build_pdf(
        arima_imgs, out_dir / "equity_arima_grid_3x2.pdf",
        cols=args.cols, rows=args.rows, cell_w=args.cell_w, cell_h=args.cell_h, padding=args.pad
    )

    # LSTM
    lstm_imgs = collect_equity_images(Path(args.lstm_dir))
    build_pdf(
        lstm_imgs, out_dir / "equity_lstm_grid_3x2.pdf",
        cols=args.cols, rows=args.rows, cell_w=args.cell_w, cell_h=args.cell_h, padding=args.pad
    )

if __name__ == "__main__":
    main()