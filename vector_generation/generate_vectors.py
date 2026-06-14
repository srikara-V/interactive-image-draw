import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.generator import (  # noqa: E402
    DEFAULT_NEGATIVE_PROMPT,
    _ddim_invert,
    _encode_image_latents,
    _encode_prompt_embeddings,
    _load_pipeline,
    _settings,
    generate_image,
    model_info,
)
from app.services.perception_vectors import VECTOR_DEFINITIONS  # noqa: E402


OUTPUT_JSON = Path(__file__).resolve().parent / "perception_vectors.json"
OUTPUT_PT = Path(__file__).resolve().parent / "perception_vectors.pt"

VECTOR_BASE_PROMPTS = [
    "premium product photo of a translucent wearable device on a workbench",
    "editorial product photograph of a compact camera module on a desk",
    "cinematic close-up of a small metal gadget on a studio table",
]


def _mean_prompt_embedding(pipeline, phrase: str) -> torch.Tensor:
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder
    device = next(text_encoder.parameters()).device
    inputs = tokenizer(
        phrase,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    with torch.no_grad():
        encoded = text_encoder(input_ids)[0]
    mask = attention_mask.unsqueeze(-1).to(encoded.dtype)
    pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return pooled[0].detach().float().cpu()


def _warmth(image: Image.Image, amount: float) -> Image.Image:
    tensor = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32))
    if amount >= 0:
        tensor[..., 0] *= 1.0 + amount * 0.22
        tensor[..., 1] *= 1.0 + amount * 0.05
        tensor[..., 2] *= 1.0 - amount * 0.16
    else:
        tensor[..., 0] *= 1.0 + amount * 0.14
        tensor[..., 1] *= 1.0 + amount * 0.03
        tensor[..., 2] *= 1.0 - amount * 0.24
    array = tensor.clamp(0, 255).byte().numpy()
    return Image.fromarray(array, mode="RGB")


def _edit_pair(name: str, image: Image.Image) -> tuple[Image.Image, Image.Image]:
    if name == "blurry":
        return image.filter(ImageFilter.GaussianBlur(radius=3.0)), image.filter(ImageFilter.UnsharpMask(radius=1.0, percent=180, threshold=2))
    if name == "contrast":
        return ImageEnhance.Contrast(image).enhance(1.85), ImageEnhance.Contrast(image).enhance(0.45)
    if name == "saturation":
        return ImageEnhance.Color(image).enhance(1.9), ImageEnhance.Color(image).enhance(0.15)
    if name == "warmth":
        return _warmth(image, 1.0), _warmth(image, -1.0)
    if name == "sharpness":
        sharp = image.filter(ImageFilter.UnsharpMask(radius=1.1, percent=230, threshold=1))
        soft = image.filter(ImageFilter.GaussianBlur(radius=1.6))
        return sharp, soft
    raise KeyError(name)


def _inverted_latent(pipeline, image: Image.Image, prompt: str) -> torch.Tensor:
    settings = _settings()
    guidance_scale = 1.2
    prompt_embeds, negative_prompt_embeds = _encode_prompt_embeddings(
        pipeline=pipeline,
        prompt=prompt,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        guidance_scale=guidance_scale,
        settings=settings,
    )
    latents = _encode_image_latents(pipeline, image, settings)
    inverted = _ddim_invert(
        pipeline=pipeline,
        latents=latents,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        guidance_scale=guidance_scale,
        steps=8,
    )
    return inverted[0].detach().float().cpu()


def _paired_edit_latent_vectors(pipeline) -> Dict[str, torch.Tensor]:
    accumulators = {name: [] for name in VECTOR_DEFINITIONS}
    for index, prompt in enumerate(VECTOR_BASE_PROMPTS):
        base = generate_image(prompt, seed=3100 + index, width=512, height=512, style="product")
        for name in VECTOR_DEFINITIONS:
            positive, negative = _edit_pair(name, base)
            positive_latent = _inverted_latent(pipeline, positive, prompt)
            negative_latent = _inverted_latent(pipeline, negative, prompt)
            direction = positive_latent - negative_latent
            if name == "sharpness":
                direction = -direction
            accumulators[name].append(direction)

    latent_vectors = {}
    for name, directions in accumulators.items():
        direction = torch.stack(directions, dim=0).mean(dim=0)
        direction = direction - direction.mean()
        norm = float(torch.linalg.vector_norm(direction).item())
        latent_vectors[name] = direction / max(norm, 1e-8)
    return latent_vectors


def generate_vectors(latent_tensors: Dict[str, torch.Tensor]) -> Dict[str, object]:
    pipeline = _load_pipeline()
    vectors = {}
    for name, definition in VECTOR_DEFINITIONS.items():
        positive = definition["positive"]
        negative = definition["negative"]
        pos = _mean_prompt_embedding(pipeline, positive)
        neg = _mean_prompt_embedding(pipeline, negative)
        direction = pos - neg
        norm = float(torch.linalg.vector_norm(direction).item())
        unit = direction / max(norm, 1e-8)
        vectors[name] = {
            "positive": positive,
            "negative": negative,
            "norm": norm,
            "dimension": int(unit.numel()),
            "embedding": [round(float(value), 7) for value in unit.tolist()],
        }
    latent_vectors = {
        name: {
            "shape": list(vector.shape),
            "norm": 1.0,
            "mean": float(vector.mean().item()),
            "std": float(vector.std().item()),
        }
        for name, vector in latent_tensors.items()
    }
    return {
        **model_info(),
        "method": "paired_edit_ddim_inversion_latent_delta",
        "vectors": vectors,
        "latent_vectors": latent_vectors,
    }


def _tensor_payload(payload: Dict[str, object], latent_tensors: Dict[str, torch.Tensor]) -> Dict[str, object]:
    vectors = payload["vectors"]
    tensor_vectors = {}
    for name, vector in vectors.items():
        tensor_vectors[name] = torch.tensor(vector["embedding"], dtype=torch.float32)
    return {
        "model_id": payload["model_id"],
        "device": payload["device"],
        "dtype": payload["dtype"],
        "method": payload["method"],
        "vectors": tensor_vectors,
        "latent_vectors": latent_tensors,
    }


def main() -> None:
    pipeline = _load_pipeline()
    latent_tensors = _paired_edit_latent_vectors(pipeline)
    payload = generate_vectors(latent_tensors)
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    torch.save(_tensor_payload(payload, latent_tensors), OUTPUT_PT)
    print("wrote %s" % OUTPUT_JSON)
    print("wrote %s" % OUTPUT_PT)
    for name, vector in payload["vectors"].items():
        latent = payload["latent_vectors"][name]
        print(
            "%s: text_dim=%s text_norm=%.4f latent_shape=%s latent_norm=%.4f"
            % (name, vector["dimension"], vector["norm"], latent["shape"], latent["norm"])
        )


if __name__ == "__main__":
    main()
