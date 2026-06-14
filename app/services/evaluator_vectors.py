import os
from functools import lru_cache
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
EVALUATOR_TENSOR_FILE = ROOT / "vector_generation" / "evaluator_vectors.pt"

PerceptionMap = Dict[str, float]


def _pick_device() -> torch.device:
    requested = os.getenv("IMAGE_EVALUATOR_DEVICE", "").strip().lower()
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluator_available() -> bool:
    return EVALUATOR_TENSOR_FILE.exists() and os.getenv("IMAGE_EVALUATOR", "clip").strip().lower() != "features"


def _slider_weights(perception: PerceptionMap) -> Dict[str, float]:
    def weight(key: str) -> float:
        return float(np.clip(float(perception.get(key, 0.0)) / 100.0, 0.0, 1.0))

    weights = {
        "blurry": weight("blurry"),
        "contrast": weight("contrast"),
        "saturation": weight("saturation"),
        "warmth": weight("warmth"),
        "sharpness": weight("sharpness"),
    }

    if weights["blurry"] >= weights["sharpness"]:
        weights["sharpness"] = 0.0
    else:
        weights["blurry"] = 0.0
    return weights


@lru_cache(maxsize=1)
def _load_clip():
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise RuntimeError("transformers is required for CLIP evaluator vectors.") from exc

    device = _pick_device()
    model_name = os.getenv("IMAGE_EVALUATOR_MODEL", "openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).eval().to(device)
    return processor, model, device


@lru_cache(maxsize=1)
def _load_vectors() -> Dict[str, torch.Tensor]:
    payload = torch.load(EVALUATOR_TENSOR_FILE, map_location="cpu", weights_only=True)
    vectors = payload.get("vectors", {})
    return {
        name: torch.nn.functional.normalize(vector.float(), dim=0)
        for name, vector in vectors.items()
        if isinstance(vector, torch.Tensor)
    }


@torch.inference_mode()
def image_embedding(image: Image.Image) -> torch.Tensor:
    processor, model, device = _load_clip()
    inputs = processor(images=[image.convert("RGB")], return_tensors="pt")
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    outputs = model.vision_model(**inputs)
    pooled = getattr(outputs, "pooler_output", None)
    if pooled is None:
        pooled = outputs[1]
    features = model.visual_projection(pooled)
    features = torch.nn.functional.normalize(features.float(), dim=1)
    return features[0].detach().cpu()


def evaluator_energy(image: Image.Image, perception: PerceptionMap) -> Dict[str, float]:
    vectors = _load_vectors()
    weights = _slider_weights(perception)
    active = {
        name: weight
        for name, weight in weights.items()
        if abs(weight) >= 0.04 and name in vectors
    }
    if not active:
        return {"embedding_energy": 0.0, "embedding_alignment": 0.0, "embedding_active_vectors": 0.0}

    direction = torch.zeros_like(next(iter(vectors.values())))
    for name, weight in active.items():
        direction = direction + float(weight) * vectors[name]
    direction = torch.nn.functional.normalize(direction, dim=0)

    embedding = image_embedding(image)
    alignment = float(torch.dot(embedding, direction).item())
    scale = float(os.getenv("IMAGE_EVALUATOR_SCALE", "4.0"))
    return {
        "embedding_energy": alignment * scale,
        "embedding_alignment": alignment,
        "embedding_active_vectors": float(len(active)),
    }


def clear_evaluator_caches() -> None:
    _load_clip.cache_clear()
    _load_vectors.cache_clear()
