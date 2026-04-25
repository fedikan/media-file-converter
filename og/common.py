"""Shared constants, fonts, and helpers for OG card rendering."""
import os
from functools import lru_cache
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONTS_DIR = os.path.join(HERE, "fonts")
ASSETS_DIR = os.path.join(HERE, "assets")

FONT_REGULAR = os.path.join(FONTS_DIR, "PublicSans-Regular.ttf")
FONT_SEMIBOLD = os.path.join(FONTS_DIR, "PublicSans-SemiBold.ttf")
FONT_BOLD = os.path.join(FONTS_DIR, "PublicSans-Bold.ttf")
LOGO_PNG = os.path.join(ASSETS_DIR, "LOGO_512.png")
LOGO_TEXT_PNG = os.path.join(ASSETS_DIR, "LOGO_TEXT.png")

# Brand palette from ropewalk-front/app/assets/styles/colors.scss (dark mode).
BG_TOP = (10, 13, 13)          # --tertiary-background
BG_BOTTOM = (3, 6, 6)          # --default-background
NODE_FILL = (24, 43, 43)       # --quaternary-background
NODE_STROKE = (22, 63, 59)     # --quaternary-stroke
STROKE = (39, 47, 45)          # --default-stroke
ACCENT = (76, 221, 206)        # --accent teal
MUTED_ACCENT = (47, 139, 129)  # --muted-accent
TEXT_PRIMARY = (218, 218, 218)   # --default-text
TEXT_SECONDARY = (148, 145, 145)  # --secondary-text
CHIP_BG = (23, 41, 41)         # --tertiary

WEIGHTS = {"regular": FONT_REGULAR, "semibold": FONT_SEMIBOLD, "bold": FONT_BOLD}


@lru_cache(maxsize=64)
def get_font(size: int, weight: str = "semibold"):
    path = WEIGHTS.get(weight, FONT_SEMIBOLD)
    if os.path.exists(path):
        return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def vertical_gradient(w: int, h: int, top, bottom) -> Image.Image:
    img = Image.new("RGB", (w, h), top)
    draw = ImageDraw.Draw(img)
    for y in range(h):
        t = y / max(h - 1, 1)
        r = int(top[0] * (1 - t) + bottom[0] * t)
        g = int(top[1] * (1 - t) + bottom[1] * t)
        b = int(top[2] * (1 - t) + bottom[2] * t)
        draw.line([(0, y), (w, y)], fill=(r, g, b))
    return img


def resize_cover(img: Image.Image, w: int, h: int) -> Image.Image:
    """Resize+crop so image exactly fills w×h (like CSS object-fit: cover)."""
    src_w, src_h = img.size
    scale = max(w / src_w, h / src_h)
    new_w = int(src_w * scale) + 1
    new_h = int(src_h * scale) + 1
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - w) // 2
    top = (new_h - h) // 2
    return resized.crop((left, top, left + w, top + h))


def round_corners(img: Image.Image, radius: int) -> Image.Image:
    """Return a copy of img with rounded corners (transparent outside)."""
    img = img.convert("RGBA")
    mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, img.width, img.height), radius=radius, fill=255)
    out = Image.new("RGBA", img.size, (0, 0, 0, 0))
    out.paste(img, (0, 0), mask=mask)
    return out


def darken(img: Image.Image, alpha: float) -> Image.Image:
    """Darken the image by overlaying a semi-transparent black layer."""
    overlay = Image.new("RGBA", img.size, (0, 0, 0, int(255 * alpha)))
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def blur_cover(img: Image.Image, w: int, h: int, blur_radius: int = 40, darkness: float = 0.5) -> Image.Image:
    cover = resize_cover(img, w, h)
    blurred = cover.filter(ImageFilter.GaussianBlur(blur_radius))
    return darken(blurred, darkness)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list:
    """Greedy word-wrap. Returns list of line strings that fit within max_width."""
    if not text:
        return []
    words = text.split()
    lines = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def clamp_lines(lines: list, max_lines: int, ellipsis: str = "…") -> list:
    """Cap lines to max_lines, appending an ellipsis to the last line if truncated."""
    if len(lines) <= max_lines:
        return lines
    kept = list(lines[:max_lines])
    kept[-1] = kept[-1].rstrip() + ellipsis
    return kept


def load_logo(target_height: int, tint=None) -> Image.Image:
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


def _tint_rgba(src: Image.Image, tint) -> Image.Image:
    """Re-color every non-transparent pixel of an RGBA image to `tint`,
    preserving the original alpha channel. Used to brand-tint monochrome
    logo assets to the accent color."""
    src = src.convert("RGBA")
    r, g, b = tint
    out = src.copy()
    pixels = out.load()
    for y in range(out.height):
        for x in range(out.width):
            _, _, _, a = pixels[x, y]
            if a > 0:
                pixels[x, y] = (r, g, b, a)
    return out


def load_wordmark(target_height: int, tint=None) -> Image.Image:
    """Load the 'ropewalk' wordmark PNG, scaled to target_height and
    optionally tinted. Returns None if the asset is missing."""
    if not os.path.exists(LOGO_TEXT_PNG):
        return None
    wm = Image.open(LOGO_TEXT_PNG).convert("RGBA")
    ratio = target_height / wm.height
    new_w = int(wm.width * ratio)
    wm = wm.resize((new_w, target_height), Image.LANCZOS)
    if tint is not None:
        wm = _tint_rgba(wm, tint)
    return wm


WHITE = (255, 255, 255)


def paste_wordmark(img: Image.Image, right_pad: int = 48, bottom_pad: int = 40):
    """Bottom-right ropewalk lockup: icon + wordmark, both rendered in white
    so the brand mark stays legible on any hero background without bleeding
    its accent color across the card.
    Prefers the real wordmark PNG; falls back to Public Sans text if missing."""
    icon_h = 42
    wm_h = 34
    gap = 12

    icon = load_logo(icon_h, tint=WHITE)
    wordmark = load_wordmark(wm_h, tint=WHITE)

    if wordmark is None:
        # Fallback: hand-rendered text.
        draw = ImageDraw.Draw(img)
        wm_font = get_font(30, "semibold")
        wm_text = "ropewalk"
        bbox = draw.textbbox((0, 0), wm_text, font=wm_font)
        wm_w = bbox[2] - bbox[0]
        if icon is not None:
            total_w = icon.width + gap + wm_w
            x_start = img.width - total_w - right_pad
            y_center = img.height - bottom_pad - icon_h
            img.paste(icon, (x_start, y_center), icon)
            text_y = y_center + (icon_h - 30) // 2 - 5
            draw.text((x_start + icon.width + gap, text_y),
                      wm_text, fill=TEXT_PRIMARY, font=wm_font)
        else:
            draw.text((img.width - wm_w - right_pad, img.height - bottom_pad - 30),
                      wm_text, fill=TEXT_PRIMARY, font=wm_font)
        return

    # Happy path: icon + raster wordmark.
    total_w = (icon.width if icon else 0) + (gap if icon else 0) + wordmark.width
    x_start = img.width - total_w - right_pad
    y_baseline = img.height - bottom_pad
    if icon is not None:
        icon_y = y_baseline - icon_h
        img.paste(icon, (x_start, icon_y), icon)
        wm_x = x_start + icon.width + gap
    else:
        wm_x = x_start
    # Vertically align wordmark's optical center with the icon
    wm_y = y_baseline - icon_h + (icon_h - wm_h) // 2 + 2
    img.paste(wordmark, (wm_x, wm_y), wordmark)


def paste_brand_badge(img: Image.Image, x: int, bottom_y: int,
                      icon_h: int = 22, text_h: int = 18):
    """Small dark-pill brand badge: white Ropewalk icon + "Ropewalk" text.
    Drawn so the badge's bottom edge sits exactly on `bottom_y` — used to
    pin the mark to the bottom line of the hero thumbnail.
    """
    draw = ImageDraw.Draw(img)
    icon = load_logo(icon_h, tint=WHITE)
    font = get_font(text_h, "bold")
    label = "Ropewalk"
    tw = draw.textlength(label, font=font)
    ascent, descent = font.getmetrics()
    glyph_h = ascent + descent

    pad_x = 12
    pad_y = 8
    gap = 8
    icon_w = icon.width if icon is not None else 0
    content_h = max(icon_h, glyph_h)
    pill_h = content_h + pad_y * 2
    pill_w = int(pad_x + icon_w + (gap if icon is not None else 0) + tw + pad_x)

    pill_x = x
    pill_y = bottom_y - pill_h
    # Dark, semi-transparent pill so the badge reads on bright images too.
    pill = Image.new("RGBA", (pill_w, pill_h), (0, 0, 0, 0))
    pdraw = ImageDraw.Draw(pill)
    pdraw.rounded_rectangle(
        (0, 0, pill_w, pill_h),
        radius=pill_h // 2,
        fill=(0, 0, 0, 180),
    )
    if icon is not None:
        icon_y_in_pill = (pill_h - icon_h) // 2
        pill.paste(icon, (pad_x, icon_y_in_pill), icon)
    # Vertically center the text via ascent so capital-height aligns with icon
    text_x_in_pill = pad_x + icon_w + (gap if icon is not None else 0)
    text_y_in_pill = (pill_h - glyph_h) // 2
    pdraw.text((text_x_in_pill, text_y_in_pill), label, fill=WHITE, font=font)
    img.paste(pill, (pill_x, pill_y), pill)
