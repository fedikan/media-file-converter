"""
Generates canvas-placeholder.png — the OG fallback image for canvas share pages
that don't yet have a captured previewUrl.

Run once locally (or via CI) and upload dist/canvas-placeholder.png to:
  https://rw-cdn.ams3.cdn.digitaloceanspaces.com/og-tags/canvas-placeholder.png

Output: 1200x630 PNG, dark-branded, ~30KB.
"""
import os
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(__file__)
W, H = 1200, 630
OUT = os.path.join(HERE, "dist", "canvas-placeholder.png")

FONTS_DIR = os.path.join(HERE, "fonts")
ASSETS_DIR = os.path.join(HERE, "assets")
FONT_REGULAR = os.path.join(FONTS_DIR, "PublicSans-Regular.ttf")
FONT_SEMIBOLD = os.path.join(FONTS_DIR, "PublicSans-SemiBold.ttf")
FONT_BOLD = os.path.join(FONTS_DIR, "PublicSans-Bold.ttf")
LOGO_PNG = os.path.join(ASSETS_DIR, "LOGO_512.png")

# Brand palette (from app/assets/styles/colors.scss — dark mode defaults)
BG_TOP = (10, 13, 13)        # --tertiary-background #0a0d0d
BG_BOTTOM = (3, 6, 6)        # --default-background  #030606
NODE_FILL = (24, 43, 43)     # --quaternary-background #182b2b
NODE_STROKE = (22, 63, 59)   # --quaternary-stroke #163F3B
STROKE = (39, 47, 45)        # --default-stroke #272F2d
ACCENT = (76, 221, 206)      # --accent #4CDDCE (teal)
MUTED_ACCENT = (47, 139, 129)  # --muted-accent #2F8B81
TEXT_PRIMARY = (218, 218, 218)   # --default-text #DADADA
TEXT_SECONDARY = (148, 145, 145) # --secondary-text #949191
CHIP_BG = (23, 41, 41)       # --tertiary #172929


def vertical_gradient(w, h, top, bottom):
    img = Image.new("RGB", (w, h), top)
    draw = ImageDraw.Draw(img)
    for y in range(h):
        t = y / max(h - 1, 1)
        r = int(top[0] * (1 - t) + bottom[0] * t)
        g = int(top[1] * (1 - t) + bottom[1] * t)
        b = int(top[2] * (1 - t) + bottom[2] * t)
        draw.line([(0, y), (w, y)], fill=(r, g, b))
    return img


def radial_glow(w, h, center, radius, color, alpha=90):
    """Subtle purple glow behind hero area."""
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    cx, cy = center
    steps = 40
    for i in range(steps, 0, -1):
        t = i / steps
        a = int(alpha * (1 - t) ** 2)
        r = int(radius * t)
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color + (a,))
    return layer.filter(ImageFilter.GaussianBlur(24))


def draw_grid(draw, w, h, spacing=60, color=(30, 30, 34)):
    for x in range(0, w, spacing):
        draw.line([(x, 0), (x, h)], fill=color, width=1)
    for y in range(0, h, spacing):
        draw.line([(0, y), (w, y)], fill=color, width=1)


def rounded_node(draw, xy, radius=14, fill=NODE_FILL, outline=NODE_STROKE, width=2):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def bezier(draw, p0, p1, color, width=2, steps=60):
    """Quadratic bezier-ish connector between two points with a mid control."""
    x0, y0 = p0
    x1, y1 = p1
    mid = ((x0 + x1) / 2, y0 + (y1 - y0) * 0.1)
    prev = p0
    for i in range(1, steps + 1):
        t = i / steps
        ox = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * mid[0] + t * t * x1
        oy = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * mid[1] + t * t * y1
        draw.line([prev, (ox, oy)], fill=color, width=width)
        prev = (ox, oy)


def load_font(size, weight="semibold"):
    weight_map = {"regular": FONT_REGULAR, "semibold": FONT_SEMIBOLD, "bold": FONT_BOLD}
    path = weight_map.get(weight, FONT_SEMIBOLD)
    if os.path.exists(path):
        return ImageFont.truetype(path, size)
    fallback = "/System/Library/Fonts/Helvetica.ttc"
    if os.path.exists(fallback):
        return ImageFont.truetype(fallback, size)
    return ImageFont.load_default()


def load_logo(target_height, tint=None):
    """Load the Ropewalk icon PNG and optionally tint it."""
    if not os.path.exists(LOGO_PNG):
        return None
    logo = Image.open(LOGO_PNG).convert("RGBA")
    ratio = target_height / logo.height
    new_w = int(logo.width * ratio)
    logo = logo.resize((new_w, target_height), Image.LANCZOS)
    if tint is not None:
        r, g, b = tint
        pixels = logo.load()
        for y in range(logo.height):
            for x in range(logo.width):
                _, _, _, a = pixels[x, y]
                if a > 0:
                    pixels[x, y] = (r, g, b, a)
    return logo


def main():
    img = vertical_gradient(W, H, BG_TOP, BG_BOTTOM)

    grid_layer = Image.new("RGB", (W, H), (0, 0, 0))
    draw_grid(ImageDraw.Draw(grid_layer), W, H, spacing=60, color=(14, 22, 22))
    img = Image.blend(img, grid_layer, 0.45)

    # Teal glow upper-left to hint at "flow"
    glow = radial_glow(W, H, center=(240, 180), radius=480, color=ACCENT, alpha=60)
    img = Image.alpha_composite(img.convert("RGBA"), glow).convert("RGB")

    draw = ImageDraw.Draw(img)

    # --- Node diagram motif (top-left area) ---
    # Three nodes + connectors, small scale
    nodes = [
        (90, 110, 230, 180),    # input node
        (310, 90, 470, 160),    # process node
        (310, 210, 470, 280),   # branch node
        (550, 150, 700, 220),   # output node
    ]
    for n in nodes:
        rounded_node(draw, n, radius=14)
        # tiny dots on ports
        draw.ellipse((n[2] - 4, (n[1] + n[3]) // 2 - 4, n[2] + 4, (n[1] + n[3]) // 2 + 4), fill=NODE_STROKE)
        draw.ellipse((n[0] - 4, (n[1] + n[3]) // 2 - 4, n[0] + 4, (n[1] + n[3]) // 2 + 4), fill=NODE_STROKE)

    # Connectors
    def right_mid(n):
        return (n[2], (n[1] + n[3]) / 2)

    def left_mid(n):
        return (n[0], (n[1] + n[3]) / 2)

    bezier(draw, right_mid(nodes[0]), left_mid(nodes[1]), ACCENT + (), width=2)
    bezier(draw, right_mid(nodes[0]), left_mid(nodes[2]), STROKE, width=2)
    bezier(draw, right_mid(nodes[1]), left_mid(nodes[3]), ACCENT, width=2)
    bezier(draw, right_mid(nodes[2]), left_mid(nodes[3]), STROKE, width=2)

    # --- Chip: CANVAS ---
    chip_font = load_font(22, "bold")
    chip_text = "CANVAS"
    bbox = draw.textbbox((0, 0), chip_text, font=chip_font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    chip_pad_x, chip_pad_y = 18, 10
    chip_w = tw + chip_pad_x * 2 + 18  # room for dot
    chip_h = th + chip_pad_y * 2 + 4
    chip_x, chip_y = 90, 360
    draw.rounded_rectangle(
        (chip_x, chip_y, chip_x + chip_w, chip_y + chip_h),
        radius=chip_h // 2,
        fill=CHIP_BG,
        outline=NODE_STROKE,
        width=1,
    )
    # small dot
    dot_r = 5
    dot_x = chip_x + 16
    dot_y = chip_y + chip_h // 2
    draw.ellipse((dot_x - dot_r, dot_y - dot_r, dot_x + dot_r, dot_y + dot_r), fill=ACCENT)
    draw.text((dot_x + 12, chip_y + chip_pad_y + 1), chip_text, fill=TEXT_PRIMARY, font=chip_font)

    # --- Headline ---
    headline_font = load_font(64, "bold")
    headline = "Visual AI workflow editor"
    draw.text((90, chip_y + chip_h + 24), headline, fill=TEXT_PRIMARY, font=headline_font)

    # --- Subhead ---
    sub_font = load_font(28, "regular")
    sub = "Connect models, images, and ideas on an infinite canvas."
    draw.text((90, chip_y + chip_h + 24 + 86), sub, fill=TEXT_SECONDARY, font=sub_font)

    # --- Wordmark bottom-right: shared lockup (icon + raster wordmark) ---
    from common import paste_wordmark as _paste_wordmark
    paste_target = img.convert("RGB")
    _paste_wordmark(paste_target, right_pad=56, bottom_pad=42)
    img = paste_target

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    img.save(OUT, "PNG", optimize=True)
    size_kb = os.path.getsize(OUT) / 1024
    print(f"wrote {OUT} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
