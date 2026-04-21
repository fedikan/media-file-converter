"""
OG card composition — produces 1200x630 branded cards for social unfurls.

Exported entry point: compose_card(data: dict) -> bytes
Supports template = 'generation' (more templates can be added later).

Keep this module fast and lightweight:
- single outbound image fetch per card (the background)
- Pillow-only compositing (no headless browser)
- WEBP q=82 output, typically 50-90KB
"""
import io
import requests
from PIL import Image, ImageDraw

from . import common as c

CARD_W, CARD_H = 1200, 630
FETCH_TIMEOUT = 5.0


def _fetch_image(url: str):
    try:
        r = requests.get(url, timeout=FETCH_TIMEOUT, stream=True)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content))
        img.load()
        return img.convert("RGB")
    except Exception:
        return None


def _draw_chip(img: Image.Image, x: int, y: int, text: str, fg=c.TEXT_PRIMARY,
               bg=c.CHIP_BG, outline=c.NODE_STROKE, dot_color=None, font_size: int = 22,
               weight: str = "bold") -> tuple:
    """Pill-shaped chip; returns the chip's bottom-right corner coordinates."""
    draw = ImageDraw.Draw(img)
    font = c.get_font(font_size, weight)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad_x, pad_y = 18, 10
    chip_h = th + pad_y * 2 + 4
    dot_lead = 22 if dot_color is not None else 0
    chip_w = tw + pad_x * 2 + dot_lead
    draw.rounded_rectangle(
        (x, y, x + chip_w, y + chip_h),
        radius=chip_h // 2,
        fill=bg,
        outline=outline,
        width=1,
    )
    if dot_color is not None:
        dot_r = 5
        dot_x = x + 16
        dot_y = y + chip_h // 2
        draw.ellipse((dot_x - dot_r, dot_y - dot_r, dot_x + dot_r, dot_y + dot_r), fill=dot_color)
        text_x = dot_x + 12
    else:
        text_x = x + pad_x
    draw.text((text_x, y + pad_y + 1), text, fill=fg, font=font)
    return x + chip_w, y + chip_h


def _render_text_block(img: Image.Image, text: str, x: int, y: int, width: int,
                       font_size: int = 36, line_height: int = 48, max_lines: int = 4,
                       weight: str = "semibold", fill=c.TEXT_PRIMARY) -> int:
    """Render wrapped+clamped text. Returns y where the block ends."""
    draw = ImageDraw.Draw(img)
    font = c.get_font(font_size, weight)
    lines = c.clamp_lines(c.wrap_text(draw, text or "", font, width), max_lines)
    cy = y
    for line in lines:
        draw.text((x, cy), line, fill=fill, font=font)
        cy += line_height
    return cy


def render_generation_card(data: dict) -> bytes:
    """
    Required fields in `data`:
      - background_url: str (URL of the actual generated image — used as hero)
      - prompt: str
      - username: str (without @)
      - model_label: str
      - is_agent_run: bool (default False)
    Optional:
      - model_icon_url: str
    """
    background_url = data.get("background_url") or ""
    prompt = (data.get("prompt") or "").strip()
    username = (data.get("username") or "").strip()
    model_label = (data.get("model_label") or "").strip()
    is_agent_run = bool(data.get("is_agent_run"))

    # --- Base: blurred hero as background; plain gradient if fetch fails.
    bg = _fetch_image(background_url)
    if bg is not None:
        canvas = c.blur_cover(bg, CARD_W, CARD_H, blur_radius=44, darkness=0.58)
    else:
        canvas = c.vertical_gradient(CARD_W, CARD_H, c.BG_TOP, c.BG_BOTTOM)
    canvas = canvas.convert("RGBA")

    # --- Left hero thumbnail (crisp, rounded) ---
    thumb_size = 520
    thumb_x, thumb_y = 56, (CARD_H - thumb_size) // 2
    if bg is not None:
        thumb = c.resize_cover(bg, thumb_size, thumb_size)
        thumb_rounded = c.round_corners(thumb, 28)
        # subtle dark ring via a slightly larger mask behind it
        ring = Image.new("RGBA", (thumb_size + 4, thumb_size + 4), (0, 0, 0, 120))
        ring = c.round_corners(ring, 30)
        canvas.paste(ring, (thumb_x - 2, thumb_y - 2), ring)
        canvas.paste(thumb_rounded, (thumb_x, thumb_y), thumb_rounded)

    # --- Right text column ---
    col_x = thumb_x + thumb_size + 56
    col_w = CARD_W - col_x - 56

    # Author chip (top)
    chip_y = thumb_y + 8
    if is_agent_run:
        chip_text = f"AGENT · @{username}" if username else "AGENT RUN"
        dot = c.ACCENT
    else:
        chip_text = f"@{username}" if username else "Ropewalk"
        dot = c.ACCENT
    _, chip_bottom = _draw_chip(canvas, col_x, chip_y, chip_text, dot_color=dot)

    # Prompt — the main message, clamp at 4 lines of 36px
    prompt_y = chip_bottom + 28
    prompt_end_y = _render_text_block(
        canvas, prompt, col_x, prompt_y, col_w,
        font_size=36, line_height=48, max_lines=4,
        weight="semibold", fill=c.TEXT_PRIMARY,
    )

    # Model chip (bottom of column)
    if model_label:
        model_chip_y = thumb_y + thumb_size - 48
        _draw_chip(
            canvas, col_x, model_chip_y,
            model_label, dot_color=c.MUTED_ACCENT,
            bg=c.CHIP_BG, font_size=20, weight="semibold",
        )

    # --- Wordmark bottom-right ---
    final = canvas.convert("RGB")
    c.paste_wordmark(final, right_pad=40, bottom_pad=32)

    out = io.BytesIO()
    final.save(out, "WEBP", quality=82, method=6)
    return out.getvalue()


def compose_card(data: dict) -> bytes:
    template = (data or {}).get("template") or ""
    payload = (data or {}).get("data") or {}
    if template == "generation":
        return render_generation_card(payload)
    raise ValueError(f"Unknown OG card template: {template!r}")
