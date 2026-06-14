import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.evaluator_vectors import _load_clip, image_embedding  # noqa: E402
from app.services.generator import generate_image, model_info  # noqa: E402
from app.services.perception_vectors import VECTOR_DEFINITIONS  # noqa: E402

OUTPUT_PT = Path(__file__).resolve().parent / "evaluator_vectors.pt"
OUTPUT_JSON = Path(__file__).resolve().parent / "evaluator_vectors.json"
EXAMPLES_DIR = Path(__file__).resolve().parent / "evaluator_examples"

BASE_PROMPTS = [
    "premium product photo of a translucent wearable device on a workbench",
    "cinematic product shot of a compact camera module on a desk",
    "editorial product photograph of a small metal gadget on a studio table",
    "macro photograph of a futuristic sensor device on a clean surface",
]


def _warmth(image: Image.Image, amount: float) -> Image.Image:
    pixels = torch.tensor(list(image.convert("RGB").getdata()), dtype=torch.float32).reshape(image.height, image.width, 3)
    pixels[..., 0] *= 1.0 + amount * 0.26
    pixels[..., 1] *= 1.0 + amount * 0.04
    pixels[..., 2] *= 1.0 - amount * 0.22
    return Image.fromarray(pixels.clamp(0, 255).byte().numpy(), mode="RGB")


def _edit_pair(name: str, image: Image.Image) -> Tuple[Image.Image, Image.Image]:
    if name == "blurry":
        positive = image.filter(ImageFilter.GaussianBlur(radius=7.0))
        negative = image.filter(ImageFilter.UnsharpMask(radius=1.0, percent=260, threshold=1))
        return positive, negative
    if name == "contrast":
        return ImageEnhance.Contrast(image).enhance(2.15), ImageEnhance.Contrast(image).enhance(0.32)
    if name == "saturation":
        return ImageEnhance.Color(image).enhance(2.25), ImageEnhance.Color(image).enhance(0.05)
    if name == "warmth":
        return _warmth(image, 1.0), _warmth(image, -1.0)
    if name == "sharpness":
        positive = image.filter(ImageFilter.UnsharpMask(radius=1.0, percent=300, threshold=1))
        negative = image.filter(ImageFilter.GaussianBlur(radius=3.2))
        return positive, negative
    raise KeyError(name)


def _mean(vectors: List[torch.Tensor]) -> torch.Tensor:
    vector = torch.stack(vectors, dim=0).mean(dim=0)
    return torch.nn.functional.normalize(vector.float(), dim=0)


def main() -> None:
    _load_clip()
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    positives: Dict[str, List[torch.Tensor]] = {name: [] for name in VECTOR_DEFINITIONS}
    negatives: Dict[str, List[torch.Tensor]] = {name: [] for name in VECTOR_DEFINITIONS}

    for index, prompt in enumerate(BASE_PROMPTS):
        base = generate_image(prompt, seed=7100 + index, width=512, height=512, style="product")
        base.save(EXAMPLES_DIR / f"base_{index}.png")
        for name in VECTOR_DEFINITIONS:
            positive, negative = _edit_pair(name, base)
            positive.save(EXAMPLES_DIR / f"{name}_{index}_positive.png")
            negative.save(EXAMPLES_DIR / f"{name}_{index}_negative.png")
            positives[name].append(image_embedding(positive))
            negatives[name].append(image_embedding(negative))

    vectors = {}
    audit = {}
    for name in VECTOR_DEFINITIONS:
        positive_mean = _mean(positives[name])
        negative_mean = _mean(negatives[name])
        direction = torch.nn.functional.normalize(positive_mean - negative_mean, dim=0)
        margin = float(torch.dot(positive_mean, direction).item() - torch.dot(negative_mean, direction).item())
        vectors[name] = direction.cpu()
        audit[name] = {
            "positive_examples": len(positives[name]),
            "negative_examples": len(negatives[name]),
            "dimension": int(direction.numel()),
            "margin": margin,
        }

    payload = {
        "method": "clip_image_mean_positive_minus_negative",
        "clip_model": os.getenv("IMAGE_EVALUATOR_MODEL", "openai/clip-vit-base-patch32"),
        "diffusion_model": model_info()["model_id"],
        "vectors": vectors,
    }
    torch.save(payload, OUTPUT_PT)
    OUTPUT_JSON.write_text(
        json.dumps(
            {
                "method": payload["method"],
                "clip_model": payload["clip_model"],
                "diffusion_model": payload["diffusion_model"],
                "vectors": audit,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {OUTPUT_PT}")
    print(f"wrote {OUTPUT_JSON}")
    for name, item in audit.items():
        print(f"{name}: dim={item['dimension']} margin={item['margin']:.4f}")


if __name__ == "__main__":
    main()
