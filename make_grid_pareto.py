from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import re, math

BASE_DIR = Path("figures/pareto_best_vs_knee")
OUT_DIR  = Path("grids_out_pdf")

ROWS, COLS = 3, 2
PER_PAGE = ROWS * COLS
CELL_W, CELL_H = 600, 400
PADDING = 20
TITLE_SIZE = 26
CAPTION_SIZE = 18
TITLE_TEXT = "{model}"   

def load_font(size):
    try:
        return ImageFont.truetype("Arial.ttf", size)
    except:
        return ImageFont.load_default()

def gather_pngs(model_dir: Path):
    pngs = []
    for fold_dir in sorted(model_dir.glob("fold_*")):
        pngs.extend(sorted(fold_dir.glob("*.png")))
    return pngs

def make_pdf(model_name: str, pngs, out_prefix: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    title_font   = load_font(TITLE_SIZE)
    caption_font = load_font(CAPTION_SIZE)

    pages = []
    num_pages = math.ceil(len(pngs) / PER_PAGE)
    for p in range(num_pages):
        batch = pngs[p*PER_PAGE:(p+1)*PER_PAGE]

        grid_w = COLS*CELL_W + (COLS+1)*PADDING
        grid_h = ROWS*CELL_H + (ROWS+1)*PADDING
        title_h   = TITLE_SIZE + 2*PADDING
        caption_h = CAPTION_SIZE + 2*PADDING

        W = grid_w
        H = title_h + grid_h + caption_h

        canvas = Image.new("RGB", (W, H), "white")
        draw = ImageDraw.Draw(canvas)


        draw.text((PADDING, PADDING),
                  TITLE_TEXT.format(model=model_name),
                  fill="black", font=title_font)


        for i, png in enumerate(batch):
            r, c = divmod(i, COLS)
            x = PADDING + c*(CELL_W + PADDING)
            y = title_h + r*(CELL_H + PADDING)
            box_w, box_h = CELL_W, CELL_H

            try:
                img = Image.open(png).convert("RGB")
                thumb = img.copy()
                thumb.thumbnail((box_w, box_h), Image.LANCZOS)
                ox = x + (box_w - thumb.width)//2
                oy = y + (box_h - thumb.height)//2
                canvas.paste(thumb, (ox, oy))
            except Exception:
                draw.rectangle([x, y, x+box_w, y+box_h], outline="gray", width=2)
                draw.text((x+10, y+10), "Missing", fill="gray", font=caption_font)

        pages.append(canvas)


    if pages:
        pdf_path = OUT_DIR / f"{out_prefix}.pdf"
        pages[0].save(pdf_path, save_all=True, append_images=pages[1:], resolution=150)
        print("Saved PDF:", pdf_path)

def main():
    # LSTM
    lstm_pngs  = gather_pngs(BASE_DIR / "lstm")
    make_pdf("LSTM", lstm_pngs, "lstm_best_vs_knee_grid")

    # ARIMA–GARCH
    arima_pngs = gather_pngs(BASE_DIR / "arima")
    make_pdf("ARIMA–GARCH", arima_pngs, "arima_best_vs_knee_grid")

if __name__ == "__main__":
    main()