import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
VECTOR_JSON_FILE = ROOT / "vector_generation" / "perception_vectors.json"
VECTOR_TENSOR_FILE = ROOT / "vector_generation" / "perception_vectors.pt"

VectorWeights = Dict[str, float]

VECTOR_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "blurry": {
        "positive": "soft blurry image, defocused, smooth edges, shallow indistinct details",
        "negative": "crisp sharp image, tack sharp details, clear edges",
    },
    "contrast": {
        "positive": "high contrast image, strong blacks and bright highlights, punchy tonal separation",
        "negative": "low contrast image, flat lighting, muted tonal separation",
    },
    "saturation": {
        "positive": "vibrant saturated colors, rich color intensity, vivid palette",
        "negative": "desaturated muted colors, restrained color intensity, subdued palette",
    },
    "warmth": {
        "positive": "warm color temperature, golden light, amber highlights, red and orange balance",
        "negative": "cool color temperature, blue cast, crisp cyan shadows",
    },
    "sharpness": {
        "positive": "sharp detailed image, crisp fine detail, clear edges, high microcontrast",
        "negative": "soft image, blurred details, hazy edges",
    },
}

def load_vector_metadata() -> Dict[str, object]:
    if VECTOR_JSON_FILE.exists():
        return json.loads(VECTOR_JSON_FILE.read_text())
    return {
        "model_id": "runtime-fallback",
        "vectors": VECTOR_DEFINITIONS,
    }


def slider_weights(perception: Dict[str, float]) -> VectorWeights:
    def direction(key: str) -> float:
        return max(-1.0, min((float(perception.get(key, 50.0)) - 50.0) / 50.0, 1.0))

    weights: VectorWeights = {
        "blurry": direction("blurry"),
        "contrast": direction("contrast"),
        "saturation": direction("saturation"),
        "warmth": direction("warmth"),
        "sharpness": direction("sharpness"),
    }
    if weights["sharpness"] < 0:
        weights["blurry"] = max(weights["blurry"], -weights["sharpness"])
        weights["sharpness"] = 0.0
    if weights["blurry"] < 0:
        weights["sharpness"] = max(weights["sharpness"], -weights["blurry"])
        weights["blurry"] = 0.0
    return weights


def load_vector_tensors(device: str, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    if not VECTOR_TENSOR_FILE.exists():
        raise FileNotFoundError(
            "Missing %s. Run `python vector_generation/generate_vectors.py` first." % VECTOR_TENSOR_FILE
        )
    payload = torch.load(VECTOR_TENSOR_FILE, map_location=device, weights_only=True)
    raw_vectors = payload.get("latent_vectors") or payload.get("vectors", {})
    return {
        name: tensor.to(device=device, dtype=dtype)
        for name, tensor in raw_vectors.items()
        if isinstance(tensor, torch.Tensor)
    }


def steering_vector(
    perception: Dict[str, float],
    device: str,
    dtype: torch.dtype,
    latent_shape: torch.Size,
    threshold: float = 0.04,
) -> torch.Tensor:
    weights = slider_weights(perception)
    vectors = load_vector_tensors(device=device, dtype=dtype)
    if not vectors:
        raise ValueError("No perception vectors are available in %s." % VECTOR_TENSOR_FILE)
    combined = torch.zeros(latent_shape, device=device, dtype=dtype)
    for name, weight in weights.items():
        if abs(weight) < threshold:
            continue
        vector = vectors.get(name)
        if vector is not None:
            if vector.ndim == 3:
                vector = vector.unsqueeze(0)
            if vector.shape[-2:] != latent_shape[-2:]:
                vector = F.interpolate(vector, size=latent_shape[-2:], mode="bilinear", align_corners=False)
            if vector.shape != latent_shape:
                vector = vector.expand(latent_shape)
            combined = combined + float(weight) * vector
    return combined
