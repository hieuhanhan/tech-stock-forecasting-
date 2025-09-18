# make_grid_cumret_3x2_pdf.py
# Python 3.8+ ; pip install pillow

from pathlib import Path
from PIL import Image
import re, argparse, sys

# ---------- helpers ----------
FOLDER_RE = re.compile(r"^fold_(\d+)_(\d+)$")

def parse_fold_interval(name: str):
    m = FOLDER_RE.match(name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))   # (fold, interval)

def collect_images(base_dir: Path, img_name: str):
    """
    Quét cumret_outputs/<model>/fold_{id}_{interval}/{img_name}
    Trả về list Path đã sắp theo (fold, interval).
    """
    if not base_dir.exists():
        print(f"[WARN] Not found: {base_dir}", file=sys.stderr)
        return []
    pairs = []
    for d in base_dir.iterdir():
        if not d.is_dir():
            continue
        key = parse_fold_interval(d.name)
        if not key:
            continue
        p = d / img_name
        if p.exists():
            pairs.append((key, p))
    pairs.sort(key=lambda t: (t[0][0], t[0][1]))  # sort by fold, then interval
    return [p for _, p in pairs]

def paste_centered(canvas: Image.Image, img_path: Path, x, y, w, h):
    try:
        im = Image.open(img_path).convert("RGB")
    except Exception:
        return
    im.thumbnail((w, h), Image.LANCZOS)
    ox = x + (w - im.width)//2
    oy = y + (h - im.height)//2
    canvas.paste(im, (ox, oy))

def build_pdf(images, out_pdf: Path, cols=2, rows=3,
              cell_w=720, cell_h=440, pad=28, bg="white"):
    """
    Ghép ảnh thành nhiều trang PDF; mỗi trang rows×cols.
    """
    if not images:
        print(f"[WARN] No images -> skip {out_pdf.name}")
        return

    per_page = rows * cols
    W = cols*cell_w + (cols+1)*pad
    H = rows*cell_h + (rows+1)*pad

    pages = []
    for i in range(0, len(images), per_page):
        batch = images[i:i+per_page]
        page = Image.new("RGB", (W, H), color=bg)
        for j, img in enumerate(batch):
            r, c = divmod(j, cols)
            x = pad + c*(cell_w + pad)
            y = pad + r*(cell_h + pad)
            paste_centered(page, img, x, y, cell_w, cell_h)
        pages.append(page)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(out_pdf, save_all=True, append_images=pages[1:])
    print(f"[OK] Saved -> {out_pdf}")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(
        description="Make 3x2 PDF grids from cumret_outputs/<model>/fold_*_*/cumulative_return.png")
    ap.add_argument("--base-dir", default="cumret_outputs", type=str,
                    help="Thư mục gốc chứa arima/ và/hoặc lstm/")
    ap.add_argument("--models", nargs="+", default=["arima", "lstm"],
                    help="Danh sách model subfolders cần ghép")
    ap.add_argument("--img-name", default="cumulative_return.png", type=str)
    ap.add_argument("--out-dir", default="figures/grids_cumret", type=str)
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--cell-w", type=int, default=720)
    ap.add_argument("--cell-h", type=int, default=440)
    ap.add_argument("--pad",    type=int, default=28)
    args = ap.parse_args()

    base = Path(args.base_dir)
    out_dir = Path(args.out_dir)

    for m in args.models:
        imgs = collect_images(base / m, args.img_name)
        build_pdf(
            imgs,
            out_dir / f"cumret_{m}_grid_{args.rows}x{args.cols}.pdf",
            cols=args.cols, rows=args.rows,
            cell_w=args.cell_w, cell_h=args.cell_h, pad=args.pad
        )

if __name__ == "__main__":
    main()