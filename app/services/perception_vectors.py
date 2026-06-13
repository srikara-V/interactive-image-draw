import json
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[2]
VECTOR_FILE = ROOT / "vector_generation" / "perception_vectors.json"

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

RUNTIME_PHRASES: Dict[str, Dict[str, str]] = {
    "blurry": {"positive": "soft blur", "negative": "crisp detail"},
    "contrast": {"positive": "high contrast", "negative": "low contrast"},
    "saturation": {"positive": "vibrant color", "negative": "muted color"},
    "warmth": {"positive": "warm golden color", "negative": "cool blue color"},
    "sharpness": {"positive": "sharp fine detail", "negative": "soft detail"},
}


def load_vector_metadata() -> Dict[str, object]:
    if VECTOR_FILE.exists():
        return json.loads(VECTOR_FILE.read_text())
    return {
        "model_id": "runtime-fallback",
        "vectors": VECTOR_DEFINITIONS,
    }


def slider_weights(perception: Dict[str, float]) -> VectorWeights:
    def direction(key: str) -> float:
        return max(-1.0, min((float(perception.get(key, 50.0)) - 50.0) / 50.0, 1.0))

    weights: VectorWeights = {
        "contrast": direction("contrast"),
        "saturation": direction("saturation"),
        "warmth": direction("warmth"),
        "sharpness": direction("sharpness"),
    }
    if weights["sharpness"] < 0:
        weights["blurry"] = -weights["sharpness"]
        weights["sharpness"] = 0.0
    return weights


def weighted_refinement_phrases(perception: Dict[str, float], threshold: float = 0.08) -> Tuple[List[str], List[str], VectorWeights]:
    metadata = load_vector_metadata()
    vectors = metadata.get("vectors", VECTOR_DEFINITIONS)
    weights = slider_weights(perception)
    positive: List[str] = []
    negative: List[str] = []

    for name, weight in weights.items():
        if abs(weight) < threshold:
            continue
        definition = RUNTIME_PHRASES.get(name) or vectors.get(name, VECTOR_DEFINITIONS.get(name, {}))
        if not isinstance(definition, dict):
            definition = RUNTIME_PHRASES.get(name, VECTOR_DEFINITIONS.get(name, {}))
        pos = str(definition.get("positive", ""))
        neg = str(definition.get("negative", ""))
        if weight > 0:
            positive.append(_weighted_phrase(pos, weight))
            if neg:
                negative.append(_weighted_phrase(neg, weight * 0.6))
        else:
            positive.append(_weighted_phrase(neg, abs(weight)))
            if pos:
                negative.append(_weighted_phrase(pos, abs(weight) * 0.6))

    return positive, negative, weights


def _weighted_phrase(phrase: str, weight: float) -> str:
    strength = max(0.05, min(abs(weight), 1.0))
    if strength >= 0.75:
        prefix = "strong"
    elif strength >= 0.38:
        prefix = "moderate"
    else:
        prefix = "subtle"
    return "%s %s" % (prefix, phrase)
