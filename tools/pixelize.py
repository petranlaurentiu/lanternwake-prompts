"""Turn an AI-generated "pixel art" sheet into clean game sprites.

Image generators fake pixel art: the grid is uneven, edges are soft and there are
hundreds of colours. This script:
  1. removes the background (flat #FF00FF magenta and its tinted fringe, or transparency),
  2. resamples onto a real grid (each output pixel = the dominant colour of its cell),
  3. snaps every pixel to one palette, shared across sheets via --palette
     (the first sheet creates it; later sheets reuse it, and --extend adds a few
     colours for hues the palette doesn't cover yet),
  4. splits the sheet into separate objects (connected shapes; sparks and fireflies
     join the nearest one), reading rows top to bottom and left to right,
  5. saves one PNG per object, or one strip per animation, plus an enlarged preview.

Names map objects in reading order: "lamp" is one sprite, "walk*6" packs the next six
objects into a horizontal strip of equal frames (feet on one line, centred on each
frame's body so the animation doesn't wobble), "-" skips an object.

Usage:
  python tools/pixelize.py SHEET --cell 9 --palette assets/sprites/palette.png \
      --names keeper*8 --out assets/sprites
  python tools/pixelize.py LAYER.png --cell 4 --whole far_hills --mirror-tile ...
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def background_mask(img: Image.Image) -> np.ndarray:
    """True where a pixel is background: transparent, or magenta / magenta-tinted fringe."""
    rgba = np.asarray(img.convert("RGBA")).astype(np.int32)
    if (rgba[..., 3] < 128).any():
        return rgba[..., 3] < 128
    r, g, b = rgba[..., 0], rgba[..., 1], rgba[..., 2]
    pure = (r > 180) & (b > 180) & (g < 110)
    # Fringe: red and blue both well above green, and close to each other.
    fringe = (r - g > 90) & (b - g > 90) & (np.abs(r - b) < 70)
    return pure | fringe


def despill(rgb: np.ndarray, bg: np.ndarray, radius: int = 5):
    """Remove the magenta blended into edge pixels (thin branches, leaf tips): near the
    background, strip the part of each pixel that is "red and blue above green", and
    drop pixels that are mostly magenta. Returns the cleaned image and background mask."""
    near = bg.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            near |= np.roll(np.roll(bg, dy, axis=0), dx, axis=1)
    edge = near & ~bg
    out = rgb.astype(np.int32)
    r, g, b = out[..., 0], out[..., 1], out[..., 2]
    spill = np.minimum(r, b) - g
    magenta_ish = edge & (spill > 50) & (np.abs(r - b) < 80)
    fix = edge & (spill > 0)
    out[..., 0] = np.where(fix, r - spill, r)
    out[..., 2] = np.where(fix, b - spill, b)
    return out.clip(0, 255).astype(np.uint8), bg | magenta_ish


def resample(rgb: np.ndarray, bg: np.ndarray, cell: float):
    """Each output pixel takes the most common foreground colour in its cell,
    or is transparent if the cell is mostly background."""
    h, w = bg.shape
    out_h, out_w = int(round(h / cell)), int(round(w / cell))
    colors = np.zeros((out_h, out_w, 3), np.uint8)
    alpha = np.zeros((out_h, out_w), bool)
    # Ignore each cell's border: that's where the fake grid blurs.
    m = max(1, int(cell * 0.2)) if cell >= 3 else 0
    for y in range(out_h):
        y0, y1 = int(y * cell), max(int((y + 1) * cell), int(y * cell) + 1)
        iy0, iy1 = y0 + m, max(y1 - m, y0 + m + 1)
        for x in range(out_w):
            x0, x1 = int(x * cell), max(int((x + 1) * cell), int(x * cell) + 1)
            ix0, ix1 = x0 + m, max(x1 - m, x0 + m + 1)
            cell_bg = bg[iy0:iy1, ix0:ix1]
            if cell_bg.size == 0 or cell_bg.mean() > 0.5:
                continue
            px = rgb[iy0:iy1, ix0:ix1][~cell_bg]
            # Dominant colour: bucket to 4 bits per channel, then average the biggest bucket.
            keys = (px >> 4).astype(np.int32)
            keys = keys[:, 0] * 256 + keys[:, 1] * 16 + keys[:, 2]
            vals, counts = np.unique(keys, return_counts=True)
            colors[y, x] = px[keys == vals[counts.argmax()]].mean(axis=0)
            alpha[y, x] = True
    return colors, alpha


def make_palette(pixels: np.ndarray, n: int) -> np.ndarray:
    """`n` colours that best cover `pixels` (octree + k-means keeps hues true)."""
    pal = Image.fromarray(pixels.reshape(1, -1, 3), "RGB").quantize(
        n, method=Image.Quantize.FASTOCTREE, kmeans=3, dither=Image.Dither.NONE)
    used = np.unique(np.asarray(pal))
    flat = pal.getpalette()[: 3 * 256]
    return np.array([flat[3 * i : 3 * i + 3] for i in used], np.uint8)


def nearest(colors: np.ndarray, palette: np.ndarray):
    """Index of and distance to the nearest palette colour, per pixel."""
    diff = colors[..., None, :].astype(np.int32) - palette.astype(np.int32)
    dist = (diff ** 2).sum(axis=-1)
    index = dist.argmin(axis=-1)
    return index, np.sqrt(np.take_along_axis(dist, index[..., None], axis=-1)[..., 0])


def extend_palette(palette: np.ndarray, pixels: np.ndarray, budget: int, tolerance: float) -> np.ndarray:
    """Add up to `budget` colours for pixels the palette fits badly (new hues)."""
    _, dist = nearest(pixels, palette)
    misfits = pixels[dist > tolerance]
    if budget <= 0 or len(misfits) < 8:
        return palette
    extra = make_palette(misfits, min(budget, len(misfits)))
    return np.concatenate([palette, extra])


def load_palette(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB")).reshape(-1, 3)


def save_palette(palette: np.ndarray, path: Path) -> None:
    # Sorted dark to light so the swatch strip is readable.
    order = np.argsort(palette.astype(np.int32).sum(axis=1), kind="stable")
    Image.fromarray(palette[order].reshape(1, -1, 3), "RGB").save(path)


def components(alpha: np.ndarray, reach: int = 1):
    """Groups of filled pixels; pixels up to `reach` apart count as touching."""
    h, w = alpha.shape
    label = np.full(alpha.shape, -1, np.int32)
    groups = []
    for y in range(h):
        for x in range(w):
            if not alpha[y, x] or label[y, x] >= 0:
                continue
            stack, group = [(y, x)], []
            label[y, x] = len(groups)
            while stack:
                cy, cx = stack.pop()
                group.append((cy, cx))
                for ny in range(max(cy - reach, 0), min(cy + reach + 1, h)):
                    for nx in range(max(cx - reach, 0), min(cx + reach + 1, w)):
                        if alpha[ny, nx] and label[ny, nx] < 0:
                            label[ny, nx] = len(groups)
                            stack.append((ny, nx))
            groups.append(np.array(group))
    return groups


class Piece:
    """One object cut from the sheet: its mask, bounds and the row baseline it stands on."""

    def __init__(self, mask: np.ndarray):
        self.mask = mask
        ys, xs = np.where(mask)
        self.top, self.bottom = int(ys.min()), int(ys.max())
        self.left, self.right = int(xs.min()), int(xs.max())
        # Anchor on the head and shoulders (the top 12-40% of the figure): they stay steady
        # through a walk cycle, while legs, cloaks, staffs and lanterns swing about.
        height = self.bottom - self.top + 1
        band = (ys >= self.top + height * 0.12) & (ys <= self.top + height * 0.40)
        self.center = float(np.median(xs[band])) if band.any() else float(np.median(xs))
        self.baseline = self.bottom


def split_grid(alpha: np.ndarray, columns: int, rows: int, min_cell: int = 20):
    """Objects from an even grid: each cell is one object (an animation frame, an icon),
    however many loose sparks or particles it has. Each shape belongs to the cell its centre
    falls in, so a lunge or a spray that overhangs its cell isn't sliced off. Empty cells are
    skipped. Each row shares a baseline."""
    h, w = alpha.shape
    cells = {}
    for group in components(alpha):
        cy, cx = group.mean(axis=0)
        key = (min(int(cy * rows / h), rows - 1), min(int(cx * columns / w), columns - 1))
        cells.setdefault(key, []).append(group)
    # Stray specks that drifted into an otherwise empty cell aren't frames.
    sizes = {key: sum(len(g) for g in groups) for key, groups in cells.items()}
    floor_size = max(min_cell, 0.04 * float(np.median(list(sizes.values())))) if min_cell else 0
    ordered = []
    for r in range(rows):
        row = []
        for c in range(columns):
            if (r, c) not in cells or sizes[(r, c)] < floor_size:
                continue
            mask = np.zeros_like(alpha)
            pts = np.concatenate(cells[(r, c)])
            mask[pts[:, 0], pts[:, 1]] = True
            row.append(Piece(mask))
        if row:
            baseline = max(p.bottom for p in row)
            for p in row:
                p.baseline = baseline
        ordered.extend(row)
    return ordered


def split(alpha: np.ndarray, min_size: int):
    """Objects in reading order. Small floating bits join the nearest big object;
    single stray pixels are dropped. Objects whose bottoms are close form a row
    and share its baseline."""
    groups = components(alpha)
    big = [g for g in groups if len(g) >= min_size]
    owners = [[g] for g in big]
    for g in groups:
        if len(g) >= min_size or len(g) < 2:
            continue
        c = g.mean(axis=0)
        nearest_big = min(range(len(big)), key=lambda i: np.abs(big[i] - c).sum(axis=1).min())
        owners[nearest_big].append(g)
    pieces = []
    for parts in owners:
        mask = np.zeros_like(alpha)
        pts = np.concatenate(parts)
        mask[pts[:, 0], pts[:, 1]] = True
        pieces.append(Piece(mask))
    # Rows: sort by bottom; a new row starts where the gap is bigger than a third of the
    # shorter object's height.
    pieces.sort(key=lambda p: p.bottom)
    rows, row = [], []
    for p in pieces:
        if row and p.bottom - row[-1].bottom > (p.bottom - p.top) / 3:
            rows.append(row)
            row = []
        row.append(p)
    if row:
        rows.append(row)
    ordered = []
    for row in rows:
        baseline = max(p.bottom for p in row)
        for p in row:
            p.baseline = baseline
        ordered.extend(sorted(row, key=lambda p: p.left))
    return ordered


def cut(colors: np.ndarray, piece: Piece, left: int, right: int, top: int) -> np.ndarray:
    """RGBA of the piece in columns left..right (may run past the sheet), rows top..baseline."""
    h = piece.baseline - top + 1
    w = right - left + 1
    out = np.zeros((h, w, 4), np.uint8)
    sx0, sx1 = max(left, 0), min(right, colors.shape[1] - 1)
    mask = piece.mask[top : piece.baseline + 1, sx0 : sx1 + 1]
    region = out[:, sx0 - left : sx1 - left + 1]
    region[..., :3] = colors[top : piece.baseline + 1, sx0 : sx1 + 1]
    region[..., 3] = mask * 255
    return out


def single(colors: np.ndarray, piece: Piece) -> Image.Image:
    return Image.fromarray(cut(colors, piece, piece.left, piece.right, piece.top), "RGBA")


def strip(colors: np.ndarray, pieces) -> Image.Image:
    """Equal frames side by side, each centred on its body with the feet on the bottom row."""
    half = max(max(p.center - p.left, p.right - p.center) for p in pieces)
    half = int(np.ceil(half)) + 1
    height = max(p.baseline - p.top + 1 for p in pieces)
    frames = []
    for p in pieces:
        c = int(round(p.center))
        top = p.baseline - height + 1
        frame = np.zeros((height, 2 * half, 4), np.uint8)
        rgba = cut(colors, p, c - half, c + half - 1, max(top, 0))
        frame[height - rgba.shape[0] :] = rgba
        frames.append(frame)
    return Image.fromarray(np.concatenate(frames, axis=1), "RGBA")


def preview(images, path: Path, scale: int) -> None:
    pad = 4
    width = sum(i.width for i in images) + pad * (len(images) + 1)
    height = max(i.height for i in images) + pad * 2
    sheet = Image.new("RGBA", (width, height), (16, 18, 34, 255))
    x = pad
    for img in images:
        sheet.alpha_composite(img, (x, height - pad - img.height))
        x += img.width + pad
    sheet.resize((width * scale, height * scale), Image.NEAREST).save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sheet")
    ap.add_argument("--cell", type=float, required=True, help="source pixels per output pixel")
    ap.add_argument("--colors", type=int, default=32, help="palette size when creating a new palette")
    ap.add_argument("--palette", default="", help="shared palette PNG: used if it exists, created if not")
    ap.add_argument("--extend", type=int, default=0, help="add up to N colours for hues the palette lacks")
    ap.add_argument("--tolerance", type=float, default=28.0, help="RGB distance that counts as a misfit")
    ap.add_argument("--names", default="", help="names in reading order: name, name*frames, or -")
    ap.add_argument("--min-size", type=int, default=40, help="smaller shapes join the nearest object")
    ap.add_argument("--grid", default="", help="COLSxROWS: split into an even grid instead of by shape")
    ap.add_argument("--grid-min", type=int, default=20, help="grid cells with fewer pixels are stray specks")
    ap.add_argument("--despill", action="store_true", help="clean magenta from edge pixels")
    ap.add_argument("--cut", default="",
                    help="source columns to clear where objects touch: x, or x:top:bottom for part of a column")
    ap.add_argument("--whole", default="", help="save the whole sheet as one image with this name")
    ap.add_argument("--mirror-tile", action="store_true", help="with --whole: append a mirrored copy so it tiles")
    ap.add_argument("--out", default="assets/sprites")
    ap.add_argument("--preview", type=int, default=4, help="enlargement for the preview image")
    args = ap.parse_args()

    img = Image.open(args.sheet)
    bg = background_mask(img)
    for cut in [c for c in args.cut.split(",") if c]:
        x, *rows = [int(v) for v in cut.split(":")]
        top, bottom = rows if rows else (0, bg.shape[0])
        bg[top:bottom, max(x - 3, 0) : x + 4] = True
    rgb = np.asarray(img.convert("RGB"))
    if args.despill:
        rgb, bg = despill(rgb, bg)
    colors, alpha = resample(rgb, bg, args.cell)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    palette_path = Path(args.palette) if args.palette else None
    if palette_path and palette_path.exists():
        palette = load_palette(palette_path)
        grown = extend_palette(palette, colors[alpha], args.extend, args.tolerance)
        if len(grown) > len(palette):
            print(f"palette: +{len(grown) - len(palette)} colours ({len(grown)} total)")
            save_palette(grown, palette_path)
        palette = grown
    else:
        palette = make_palette(colors[alpha], args.colors)
        if palette_path:
            save_palette(palette, palette_path)
            print(f"new shared palette saved to {palette_path}")
    index, _ = nearest(colors, palette)
    colors = palette[index]

    images = []
    if args.whole:
        ys = np.where(alpha.any(axis=1))[0]
        rgba = np.zeros(alpha.shape + (4,), np.uint8)
        rgba[..., :3] = colors
        rgba[..., 3] = alpha * 255
        rgba = rgba[ys.min() : ys.max() + 1]
        if args.mirror_tile:
            rgba = np.concatenate([rgba, rgba[:, ::-1]], axis=1)
        image = Image.fromarray(rgba, "RGBA")
        image.save(out / f"{args.whole}.png")
        print(f"{args.whole}: {image.width}x{image.height}")
        images.append(image)
    else:
        if args.grid:
            columns, rows = (int(v) for v in args.grid.lower().split("x"))
            pieces = split_grid(alpha, columns, rows, args.grid_min)
        else:
            pieces = split(alpha, args.min_size)
        names = [n for n in args.names.split(",") if n]
        wanted = sum(int(n.split("*")[1]) if "*" in n else 1 for n in names)
        if names and wanted != len(pieces):
            print(f"warning: {len(pieces)} objects found, names describe {wanted}")
        i = 0
        for name in names:
            count = int(name.split("*")[1]) if "*" in name else 1
            group = pieces[i : i + count]
            i += count
            if name == "-" or not group:
                continue
            name = name.split("*")[0]
            image = strip(colors, group) if count > 1 else single(colors, group[0])
            image.save(out / f"{name}.png")
            label = f"{count} frames of {image.width // count}x{image.height}" if count > 1 else f"{image.width}x{image.height}"
            print(f"{name}: {label}")
            images.append(image)
        for j, piece in enumerate(pieces[i:]):
            image = single(colors, piece)
            image.save(out / f"_unnamed_{j}.png")
            images.append(image)
    if images:
        # Previews live in a folder Godot ignores, so they never ship with the game.
        previews = out / "_preview"
        previews.mkdir(exist_ok=True)
        (previews / ".gdignore").touch()
        preview(images, previews / f"{Path(args.sheet).stem}.png", args.preview)


if __name__ == "__main__":
    main()
