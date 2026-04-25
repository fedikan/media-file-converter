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
    """Pill-shaped chip; returns the chip's bottom-right corner coordinates.

    Layout (symmetric):
      [pad_x][dot][gap_dot_text][text][pad_x]
    """
    draw = ImageDraw.Draw(img)
    font = c.get_font(font_size, weight)
    # Use font's actual ascent/descent so vertical centering doesn't drift
    # based on the glyphs present (e.g. a chip with only uppercase letters
    # otherwise sits higher than one with descenders).
    ascent, descent = font.getmetrics()
    text_h = ascent + descent
    tw = draw.textlength(text, font=font)
    pad_x, pad_y = 20, 10
    dot_r = 5
    gap_dot_text = 10
    dot_lead = (dot_r * 2 + gap_dot_text) if dot_color is not None else 0
    chip_h = text_h + pad_y * 2
    chip_w = int(pad_x + dot_lead + tw + pad_x)
    draw.rounded_rectangle(
        (x, y, x + chip_w, y + chip_h),
        radius=chip_h // 2,
        fill=bg,
        outline=outline,
        width=1,
    )
    if dot_color is not None:
        dot_cx = x + pad_x + dot_r
        dot_cy = y + chip_h // 2
        draw.ellipse(
            (dot_cx - dot_r, dot_cy - dot_r, dot_cx + dot_r, dot_cy + dot_r),
            fill=dot_color,
        )
        text_x = x + pad_x + dot_lead
    else:
        text_x = x + pad_x
    # Baseline anchor for consistent vertical alignment across chips
    draw.text((text_x, y + pad_y), text, fill=fg, font=font)
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

    # --- Bottom-left brand badge, flush with the thumbnail's bottom edge.
    # Replaces the baked-in image watermark — the hero is now the optimised
    # (un-watermarked) file, so we reintroduce the Ropewalk mark here in a
    # controlled way and align it with the image's bottom line.
    c.paste_brand_badge(
        canvas,
        x=thumb_x + 18,
        bottom_y=thumb_y + thumb_size - 18,
    )

    # --- Right text column ---
    col_x = thumb_x + thumb_size + 56
    col_w = CARD_W - col_x - 56

    # Stack top-down: model chip → author chip → prompt.
    # Matches the in-app chat layout where the model label heads the message
    # and the author sits right above their prompt.
    stack_y = thumb_y + 8

    if model_label:
        _, stack_y = _draw_chip(
            canvas, col_x, stack_y,
            model_label, dot_color=c.ACCENT,
            bg=c.CHIP_BG, font_size=22, weight="bold",
        )
        stack_y += 12

    if is_agent_run:
        chip_text = f"AGENT · @{username}" if username else "AGENT RUN"
    else:
        chip_text = f"@{username}" if username else "Ropewalk"
    _, stack_y = _draw_chip(canvas, col_x, stack_y, chip_text, dot_color=c.ACCENT)

    # Prompt — the main message, clamp at 4 lines of 36px
    prompt_y = stack_y + 24
    _render_text_block(
        canvas, prompt, col_x, prompt_y, col_w,
        font_size=36, line_height=48, max_lines=4,
        weight="semibold", fill=c.TEXT_PRIMARY,
    )

    # --- Wordmark bottom-right ---
    final = canvas.convert("RGB")
    c.paste_wordmark(final, right_pad=40, bottom_pad=32)

    out = io.BytesIO()
    final.save(out, "WEBP", quality=82, method=6)
    return out.getvalue()


def render_model_card(data: dict) -> bytes:
    """
    Branded card for a model page (/model/:id).

    Required fields in `data`:
      - name: str (model.label)
    Optional:
      - icon_url: str (model.iconUrl — raster PNG/JPEG works; SVG falls back)
      - fallback_image_url: str (typically model.examples[0] — used if icon_url
        is SVG or unreachable, since Ropewalk icons are often SVG)
      - author: str (provider / org name)
      - description: str (short blurb; 2-line clamp)
      - model_type: str ('Image', 'Video', 'Text', 'Audio', '3D' etc)
    """
    name = (data.get("name") or "").strip()
    icon_url = (data.get("icon_url") or "").strip()
    fallback_image_url = (data.get("fallback_image_url") or "").strip()
    author = (data.get("author") or "").strip()
    description = (data.get("description") or "").strip()
    model_type = (data.get("model_type") or "").strip()

    canvas = c.vertical_gradient(CARD_W, CARD_H, c.BG_TOP, c.BG_BOTTOM).convert("RGBA")

    # Subtle teal glow behind the icon area to hint at brand.
    glow_layer = Image.new("RGBA", (CARD_W, CARD_H), (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(glow_layer)
    steps = 50
    cx, cy = 260, CARD_H // 2
    for i in range(steps, 0, -1):
        t = i / steps
        a = int(50 * (1 - t) ** 2)
        r = int(520 * t)
        gdraw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=c.ACCENT + (a,))
    from PIL import ImageFilter as _IF
    glow_layer = glow_layer.filter(_IF.GaussianBlur(28))
    canvas = Image.alpha_composite(canvas, glow_layer)

    # Icon block — big rounded square on the left. Try icon_url first; if that
    # fails (SVG / unreachable), fall back to the model's first example image,
    # then to the Ropewalk logo so the frame is never empty.
    icon_size = 280
    icon_x, icon_y = 80, (CARD_H - icon_size) // 2
    icon_img = _fetch_image(icon_url) if icon_url else None
    if icon_img is None and fallback_image_url:
        icon_img = _fetch_image(fallback_image_url)

    # Frame (always drawn, acts as a visual anchor even on the fallback paths)
    frame = Image.new("RGBA", (icon_size + 16, icon_size + 16), (0, 0, 0, 0))
    fdraw = ImageDraw.Draw(frame)
    fdraw.rounded_rectangle(
        (0, 0, frame.width, frame.height),
        radius=38,
        fill=c.NODE_FILL + (255,),
        outline=c.NODE_STROKE + (255,),
        width=2,
    )
    canvas.paste(frame, (icon_x - 8, icon_y - 8), frame)

    if icon_img is not None:
        icon_cover = c.resize_cover(icon_img, icon_size, icon_size)
        icon_rounded = c.round_corners(icon_cover, 32)
        canvas.paste(icon_rounded, (icon_x, icon_y), icon_rounded)
    else:
        # Final fallback: Ropewalk logo tinted teal, centred in the frame.
        logo = c.load_logo(int(icon_size * 0.55), tint=c.ACCENT)
        if logo is not None:
            lx = icon_x + (icon_size - logo.width) // 2
            ly = icon_y + (icon_size - logo.height) // 2
            canvas.paste(logo, (lx, ly), logo)

    # Right column — text block, vertically centered against the icon.
    col_x = icon_x + icon_size + 72
    col_w = CARD_W - col_x - 80
    draw = ImageDraw.Draw(canvas)

    # --- First pass: measure everything so we can center the block.
    CHIP_H = 44     # includes padding
    CHIP_TO_NAME = 18
    NAME_LINE_H = 80
    NAME_FONT = c.get_font(68, "bold")
    NAME_TO_BY = 8
    BY_H = 36
    BY_FONT = c.get_font(26, "regular")
    BY_TO_DESC = 22
    DESC_LINE_H = 36
    DESC_FONT = c.get_font(26, "regular")

    name_lines = c.clamp_lines(
        c.wrap_text(draw, name or "Model", NAME_FONT, col_w), 2
    )
    desc_lines = c.clamp_lines(
        c.wrap_text(draw, description, DESC_FONT, col_w), 2
    ) if description else []

    has_type = bool(model_type)
    has_author = bool(author)
    total_h = 0
    if has_type:
        total_h += CHIP_H + CHIP_TO_NAME
    total_h += NAME_LINE_H * len(name_lines)
    if has_author:
        total_h += NAME_TO_BY + BY_H
    if desc_lines:
        total_h += BY_TO_DESC + DESC_LINE_H * len(desc_lines)

    # Center vertically against the icon (icon_y .. icon_y + icon_size)
    icon_center_y = icon_y + icon_size // 2
    y = icon_center_y - total_h // 2

    # --- Second pass: actually draw.
    if has_type:
        _draw_chip(
            canvas, col_x, y,
            model_type.upper(),
            dot_color=c.ACCENT, font_size=20, weight="bold",
        )
        y += CHIP_H + CHIP_TO_NAME
    for line in name_lines:
        draw.text((col_x, y), line, fill=c.TEXT_PRIMARY, font=NAME_FONT)
        y += NAME_LINE_H
    if has_author:
        draw.text((col_x, y + NAME_TO_BY), f"by {author}", fill=c.TEXT_SECONDARY, font=BY_FONT)
        y += NAME_TO_BY + BY_H
    if desc_lines:
        y += BY_TO_DESC
        for line in desc_lines:
            draw.text((col_x, y), line, fill=c.TEXT_PRIMARY, font=DESC_FONT)
            y += DESC_LINE_H

    final = canvas.convert("RGB")
    c.paste_wordmark(final, right_pad=40, bottom_pad=32)

    out = io.BytesIO()
    final.save(out, "WEBP", quality=82, method=6)
    return out.getvalue()


def render_canvas_card(data: dict) -> bytes:
    """
    Branded card for a canvas page (/canvas/:id).

    Layout:
      - Brand gradient + faint grid background
      - Top-left "CANVAS" chip
      - Canvas preview (previewUrl) letterboxed into a rounded hero on the right
      - Bottom-left: canvas name + metadata line (node count, age, author)
      - Bottom-right: ropewalk wordmark

    Required fields in `data`:
      - name: str (canvas.name, falls back to "Untitled canvas")
    Optional:
      - preview_url: str (captured VueFlow preview — real canvas contents)
      - description: str
      - owner_username: str
      - node_count: int
      - updated_ago: str (e.g. "2d ago", prepared server-side)
    """
    name = (data.get("name") or "").strip() or "Untitled canvas"
    preview_url = (data.get("preview_url") or "").strip()
    description = (data.get("description") or "").strip()
    owner_username = (data.get("owner_username") or "").strip()
    node_count = data.get("node_count")
    updated_ago = (data.get("updated_ago") or "").strip()

    canvas = c.vertical_gradient(CARD_W, CARD_H, c.BG_TOP, c.BG_BOTTOM).convert("RGBA")

    # Faint grid for the "infinite canvas" feel
    grid = Image.new("RGBA", (CARD_W, CARD_H), (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(grid)
    for x in range(0, CARD_W, 60):
        gdraw.line([(x, 0), (x, CARD_H)], fill=(14, 22, 22, 140), width=1)
    for y in range(0, CARD_H, 60):
        gdraw.line([(0, y), (CARD_W, y)], fill=(14, 22, 22, 140), width=1)
    canvas = Image.alpha_composite(canvas, grid)

    draw = ImageDraw.Draw(canvas)

    # Layout: left text column (420w) + right hero (680w)
    LEFT_X = 56
    LEFT_W = 440
    HERO_X = 520
    HERO_W = CARD_W - HERO_X - 56  # = 624
    HERO_H = int(HERO_W * 9 / 16)   # 16:9 = 351
    HERO_Y = (CARD_H - HERO_H) // 2

    # Hero: canvas preview inside a rounded frame
    hero_img = _fetch_image(preview_url) if preview_url else None
    # frame (always drawn; panel behind preview OR as the placeholder itself)
    frame = Image.new("RGBA", (HERO_W + 12, HERO_H + 12), (0, 0, 0, 0))
    fdraw = ImageDraw.Draw(frame)
    fdraw.rounded_rectangle(
        (0, 0, frame.width, frame.height),
        radius=26, fill=c.NODE_FILL + (255,),
        outline=c.NODE_STROKE + (255,), width=2,
    )
    canvas.paste(frame, (HERO_X - 6, HERO_Y - 6), frame)
    if hero_img is not None:
        hero_cover = c.resize_cover(hero_img, HERO_W, HERO_H)
        hero_rounded = c.round_corners(hero_cover, 20)
        canvas.paste(hero_rounded, (HERO_X, HERO_Y), hero_rounded)
    else:
        # Placeholder: faint node-diagram motif hinting that the canvas is empty
        pdraw = ImageDraw.Draw(canvas)
        nodes = [
            (HERO_X + 60, HERO_Y + 90, HERO_X + 220, HERO_Y + 150),
            (HERO_X + 280, HERO_Y + 70, HERO_X + 440, HERO_Y + 130),
            (HERO_X + 280, HERO_Y + 200, HERO_X + 440, HERO_Y + 260),
            (HERO_X + 490, HERO_Y + 140, HERO_X + 580, HERO_Y + 200),
        ]
        for n in nodes:
            pdraw.rounded_rectangle(n, radius=12, outline=c.NODE_STROKE, width=2)

    # Left column — text block, vertically centered against the hero
    # Measure the block first
    CHIP_H = 44
    CHIP_TO_NAME = 22
    NAME_FONT = c.get_font(52, "bold")
    NAME_LINE_H = 60
    NAME_TO_DESC = 12
    DESC_FONT = c.get_font(22, "regular")
    DESC_LINE_H = 30
    DESC_TO_META = 22
    META_H = 28

    name_lines = c.clamp_lines(c.wrap_text(draw, name, NAME_FONT, LEFT_W), 2)
    desc_lines = c.clamp_lines(c.wrap_text(draw, description, DESC_FONT, LEFT_W), 2) \
        if description else []

    meta_bits = []
    if isinstance(node_count, int) and node_count >= 0:
        meta_bits.append(f"{node_count} node{'s' if node_count != 1 else ''}")
    if owner_username:
        meta_bits.append(f"@{owner_username}")
    if updated_ago:
        meta_bits.append(updated_ago)
    meta_line = "  ·  ".join(meta_bits)

    total_h = CHIP_H + CHIP_TO_NAME + NAME_LINE_H * len(name_lines)
    if desc_lines:
        total_h += NAME_TO_DESC + DESC_LINE_H * len(desc_lines)
    if meta_line:
        total_h += DESC_TO_META + META_H

    hero_center_y = HERO_Y + HERO_H // 2
    y = hero_center_y - total_h // 2

    # Draw chip
    _draw_chip(canvas, LEFT_X, y, "CANVAS", dot_color=c.ACCENT, font_size=20, weight="bold")
    y += CHIP_H + CHIP_TO_NAME
    # Name
    for line in name_lines:
        draw.text((LEFT_X, y), line, fill=c.TEXT_PRIMARY, font=NAME_FONT)
        y += NAME_LINE_H
    # Description
    if desc_lines:
        y += NAME_TO_DESC
        for line in desc_lines:
            draw.text((LEFT_X, y), line, fill=c.TEXT_SECONDARY, font=DESC_FONT)
            y += DESC_LINE_H
    # Meta line (nodes · @user · updated)
    if meta_line:
        y += DESC_TO_META
        meta_font = c.get_font(20, "semibold")
        draw.text((LEFT_X, y), meta_line, fill=c.TEXT_SECONDARY, font=meta_font)

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
    if template == "model":
        return render_model_card(payload)
    if template == "canvas":
        return render_canvas_card(payload)
    raise ValueError(f"Unknown OG card template: {template!r}")
