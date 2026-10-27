#!/usr/bin/env python3
"""Generate the app icon for "LLM 4 4s".

Two rules matter more than the artwork on iOS 6:

  * the PNG must have **no alpha channel**.  SpringBoard treats an icon with
    alpha as pre-rendered and skips the rounded-corner mask, which is exactly
    why the previous attempt showed up as a hard square next to the properly
    rounded stock icons.  Saving as RGB fixes it, and UIPrerenderedIcon is
    false in Info.plist.
  * iOS 6 wants 57x57 (Icon.png) and 114x114 (Icon@2x.png).

Design: a dark warm gradient field with a single bold "4" in the ember ramp,
drawn at 4x and downsampled so edges stay clean.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = Path(__file__).resolve().parent
SIZES = [("Icon.png", 57), ("Icon@2x.png", 114)]
SS = 4                                  # supersampling factor

TOP = (0x2E, 0x1D, 0x19)
BOT = (0x0D, 0x08, 0x07)
GLOWC = (0x7A, 0x2E, 0x0A)
RAMP = [(0xFF, 0xE2, 0xD6), (0xFF, 0xA8, 0x7C), (0xE8, 0x62, 0x1E), (0xB3, 0x3A, 0x06)]

FONT_CANDIDATES = [
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
]


def vertical_gradient(S, top, bot):
    img = Image.new("RGB", (S, S))
    d = ImageDraw.Draw(img)
    for y in range(S):
        t = y / max(1, S - 1)
        d.line([(0, y), (S, y)],
               fill=(int(top[0] + (bot[0] - top[0]) * t),
                     int(top[1] + (bot[1] - top[1]) * t),
                     int(top[2] + (bot[2] - top[2]) * t)))
    return img


def glyph_mask(S, ch="4", fill=0.76):
    """Render the glyph, then scale it by its ink bounds so the digit really
    fills `fill` of the canvas.  Font metrics lie about a digit's height (no
    descender, generous side bearings), so relying on the em box leaves the
    mark small and floating."""
    font = None
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            try:
                font = ImageFont.truetype(path, int(S * 0.9))
                break
            except Exception:
                continue
    if font is None:
        raise SystemExit("no usable system font found")

    big = Image.new("L", (S * 2, S * 2), 0)
    d = ImageDraw.Draw(big)
    d.text((S * 0.5, S * 0.5), ch, font=font, fill=255)
    box = big.getbbox()
    glyph = big.crop(box)

    target_h = int(S * fill)
    target_w = max(1, int(glyph.width * target_h / glyph.height))
    glyph = glyph.resize((target_w, target_h), Image.LANCZOS)

    m = Image.new("L", (S, S), 0)
    m.paste(glyph, ((S - target_w) // 2, (S - target_h) // 2 - int(S * 0.005)))
    return m


def glyph_gradient(S):
    """Vertical ember ramp used to fill the glyph."""
    img = Image.new("RGB", (S, S))
    d = ImageDraw.Draw(img)
    stops = RAMP
    for y in range(S):
        t = y / max(1, S - 1) * (len(stops) - 1)
        i = min(int(t), len(stops) - 2)
        f = t - i
        a, b = stops[i], stops[i + 1]
        d.line([(0, y), (S, y)],
               fill=(int(a[0] + (b[0] - a[0]) * f),
                     int(a[1] + (b[1] - a[1]) * f),
                     int(a[2] + (b[2] - a[2]) * f)))
    return img


def build(size):
    S = size * SS

    img = vertical_gradient(S, TOP, BOT)

    # warm halo behind the digit so it does not float on flat black
    glow = Image.new("L", (S, S), 0)
    gd = ImageDraw.Draw(glow)
    cx, cy, rr = S * 0.5, S * 0.52, S * 0.46
    gd.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=90)
    glow = glow.filter(ImageFilter.GaussianBlur(S * 0.09))
    img = Image.composite(Image.new("RGB", (S, S), GLOWC), img, glow)

    mask = glyph_mask(S)
    img = Image.composite(glyph_gradient(S), img, mask)

    out = img.resize((size, size), Image.LANCZOS)
    return out.convert("RGB")          # NO alpha channel: let iOS round it


def main():
    for name, size in SIZES:
        im = build(size)
        path = OUT / name
        im.save(path, "PNG", optimize=True)
        print(f"{name:14s} {im.size[0]}x{im.size[1]} mode={im.mode}")


if __name__ == "__main__":
    main()
