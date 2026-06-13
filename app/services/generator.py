import hashlib
import math
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


Palette = Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]


PALETTES: Dict[str, Palette] = {
    "cinematic": ((21, 23, 28), (221, 111, 72), (66, 158, 155)),
    "product": ((244, 246, 248), (41, 45, 50), (0, 128, 122)),
    "editorial": ((246, 241, 232), (178, 42, 71), (39, 88, 122)),
    "concept": ((28, 26, 36), (123, 92, 255), (255, 191, 73)),
    "abstract": ((18, 18, 20), (64, 186, 128), (218, 74, 111)),
}


def _seed_from_prompt(prompt: str, seed: int) -> int:
    digest = hashlib.blake2b(("%s:%s" % (prompt, seed)).encode("utf-8"), digest_size=8)
    return int.from_bytes(digest.digest(), "big", signed=False)


def _keyword_style(prompt: str, requested_style: str) -> str:
    if requested_style in PALETTES:
        return requested_style

    lowered = prompt.lower()
    if any(word in lowered for word in ("shoe", "watch", "bottle", "device", "phone", "product")):
        return "product"
    if any(word in lowered for word in ("film", "street", "city", "night", "cinematic")):
        return "cinematic"
    if any(word in lowered for word in ("magazine", "fashion", "editorial", "portrait")):
        return "editorial"
    if any(word in lowered for word in ("future", "robot", "space", "cyber", "concept")):
        return "concept"
    return "abstract"


def _gradient(width: int, height: int, palette: Palette, rng: np.random.Generator) -> np.ndarray:
    xs = np.linspace(0.0, 1.0, width, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)

    angle = rng.uniform(-math.pi, math.pi)
    wave = np.sin((xx * math.cos(angle) + yy * math.sin(angle)) * rng.uniform(5.0, 11.0))
    radial = np.sqrt((xx - rng.uniform(0.25, 0.75)) ** 2 + (yy - rng.uniform(0.25, 0.75)) ** 2)
    mix_a = np.clip((xx + yy + wave * 0.12) / 2.0, 0.0, 1.0)
    mix_b = np.clip(1.0 - radial * rng.uniform(1.1, 1.8), 0.0, 1.0)

    c0 = np.array(palette[0], dtype=np.float32)
    c1 = np.array(palette[1], dtype=np.float32)
    c2 = np.array(palette[2], dtype=np.float32)

    base = c0 * (1.0 - mix_a[..., None]) + c1 * mix_a[..., None]
    base = base * (1.0 - mix_b[..., None] * 0.55) + c2 * (mix_b[..., None] * 0.55)
    noise = rng.normal(0.0, 6.0, size=base.shape)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def _draw_subject(draw: ImageDraw.ImageDraw, width: int, height: int, style: str, rng: np.random.Generator) -> None:
    cx = int(width * rng.uniform(0.40, 0.60))
    cy = int(height * rng.uniform(0.38, 0.55))
    accent = tuple(int(v) for v in PALETTES[style][2])
    shadow = (0, 0, 0, 95)

    if style == "product":
        body_w = int(width * rng.uniform(0.28, 0.38))
        body_h = int(height * rng.uniform(0.34, 0.46))
        x0, y0 = cx - body_w // 2, cy - body_h // 2
        x1, y1 = cx + body_w // 2, cy + body_h // 2
        draw.rounded_rectangle((x0 + 18, y0 + 24, x1 + 18, y1 + 24), radius=32, fill=shadow)
        draw.rounded_rectangle((x0, y0, x1, y1), radius=34, fill=(246, 248, 250, 230), outline=(24, 28, 31, 180), width=4)
        draw.line((x0 + 35, y0 + 60, x1 - 35, y0 + 60), fill=accent + (210,), width=7)
        draw.ellipse((cx - 44, y1 - 95, cx + 44, y1 - 7), fill=(24, 28, 31, 230))
        return

    if style == "editorial":
        radius = int(width * 0.17)
        draw.ellipse((cx - radius + 20, cy - radius + 28, cx + radius + 20, cy + radius + 28), fill=shadow)
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(248, 224, 205, 235), outline=accent + (220,), width=5)
        draw.polygon(
            [(cx - radius, cy + radius), (cx + radius, cy + radius), (cx + int(radius * 1.4), height), (cx - int(radius * 1.4), height)],
            fill=(32, 36, 40, 225),
        )
        return

    if style == "cinematic":
        horizon = int(height * rng.uniform(0.56, 0.68))
        draw.rectangle((0, horizon, width, height), fill=(13, 16, 18, 125))
        for offset in range(-4, 5):
            x = cx + offset * int(width * 0.055)
            draw.polygon([(x, horizon), (x + int(width * 0.035), height), (x - int(width * 0.035), height)], fill=(20, 23, 24, 170))
        draw.ellipse((cx - 36, horizon - 120, cx + 36, horizon - 48), fill=accent + (220,))
        return

    for idx in range(10):
        size = int(width * rng.uniform(0.07, 0.20))
        x = int(width * rng.uniform(0.08, 0.84))
        y = int(height * rng.uniform(0.08, 0.80))
        color = tuple(int(v) for v in PALETTES[style][1 if idx % 2 else 2]) + (int(rng.uniform(82, 170)),)
        if rng.random() > 0.45:
            draw.ellipse((x, y, x + size, y + size), fill=color)
        else:
            draw.rounded_rectangle((x, y, x + size, y + int(size * rng.uniform(0.5, 1.4))), radius=16, fill=color)


def generate_image(prompt: str, seed: int = 7, width: int = 768, height: int = 768, style: str = "auto") -> Image.Image:
    style = _keyword_style(prompt, style)
    rng = np.random.default_rng(_seed_from_prompt(prompt, seed))
    width = int(np.clip(width, 384, 1024))
    height = int(np.clip(height, 384, 1024))

    base = Image.fromarray(_gradient(width, height, PALETTES[style], rng), mode="RGB")
    base = base.filter(ImageFilter.GaussianBlur(radius=1.3))

    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    _draw_subject(draw, width, height, style, rng)

    for _ in range(18):
        x0 = int(rng.uniform(-width * 0.15, width * 0.95))
        y0 = int(rng.uniform(-height * 0.15, height * 0.95))
        x1 = x0 + int(rng.uniform(width * 0.12, width * 0.42))
        y1 = y0 + int(rng.uniform(height * 0.04, height * 0.18))
        color = tuple(int(v) for v in PALETTES[style][int(rng.integers(0, 3))]) + (int(rng.uniform(18, 68)),)
        draw.rounded_rectangle((x0, y0, x1, y1), radius=int(rng.uniform(8, 28)), fill=color)

    image = Image.alpha_composite(base.convert("RGBA"), layer).convert("RGB")
    image = ImageEnhance.Contrast(image).enhance(1.08 if style != "product" else 0.96)
    image = ImageEnhance.Sharpness(image).enhance(1.08)
    return image


def available_styles() -> List[str]:
    return ["auto"] + sorted(PALETTES.keys())
