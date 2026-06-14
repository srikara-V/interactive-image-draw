import math
import os
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageFilter

from app.services.evaluator_vectors import evaluator_available, evaluator_energy
from app.services.generator import refine_image


FeatureMap = Dict[str, float]
PerceptionMap = Dict[str, float]


FEATURE_KEYS = ("brightness", "contrast", "saturation", "warmth", "sharpness", "focus", "entropy")


def _as_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _luminance(arr: np.ndarray) -> np.ndarray:
    return arr[..., 0] * 0.2126 + arr[..., 1] * 0.7152 + arr[..., 2] * 0.0722


def _entropy(lum: np.ndarray) -> float:
    hist, _ = np.histogram(lum, bins=64, range=(0.0, 1.0), density=False)
    probs = hist.astype(np.float64)
    probs = probs / max(float(probs.sum()), 1.0)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)) / 6.0)


def _gradient_strength(lum: np.ndarray) -> np.ndarray:
    gx = np.abs(np.diff(lum, axis=1, append=lum[:, -1:]))
    gy = np.abs(np.diff(lum, axis=0, append=lum[-1:, :]))
    return np.sqrt(gx * gx + gy * gy)


def image_features(image: Image.Image) -> FeatureMap:
    arr = _as_array(image)
    lum = _luminance(arr)
    gradient = _gradient_strength(lum)
    h, w = lum.shape
    y0, y1 = int(h * 0.22), int(h * 0.78)
    x0, x1 = int(w * 0.22), int(w * 0.78)
    center_detail = float(gradient[y0:y1, x0:x1].mean())
    border_mask = np.ones_like(lum, dtype=bool)
    border_mask[y0:y1, x0:x1] = False
    border_detail = float(gradient[border_mask].mean())

    max_channel = arr.max(axis=2)
    min_channel = arr.min(axis=2)
    saturation = np.divide(max_channel - min_channel, np.maximum(max_channel, 1e-6))
    warmth_raw = float(arr[..., 0].mean() - arr[..., 2].mean())

    return {
        "brightness": float(np.clip(lum.mean(), 0.0, 1.0)),
        "contrast": float(np.clip(lum.std() * 2.5, 0.0, 1.0)),
        "saturation": float(np.clip(saturation.mean(), 0.0, 1.0)),
        "warmth": float(np.clip(0.5 + warmth_raw * 0.9, 0.0, 1.0)),
        "sharpness": float(np.clip(gradient.mean() * 9.0, 0.0, 1.0)),
        "focus": float(np.clip(0.5 + (center_detail - border_detail) * 9.0, 0.0, 1.0)),
        "entropy": float(np.clip(_entropy(lum), 0.0, 1.0)),
    }


def perception_to_targets(perception: PerceptionMap, base_features: FeatureMap) -> FeatureMap:
    targets: FeatureMap = {}
    for key in FEATURE_KEYS:
        slider = float(perception.get(key, 50.0))
        slider = float(np.clip(slider, 0.0, 100.0))
        neutral = base_features.get(key, 0.5)
        targets[key] = float(np.clip(neutral + ((slider - 50.0) / 50.0) * 0.38, 0.02, 0.98))
    blurry = float(np.clip(perception.get("blurry", 50.0), 0.0, 100.0))
    if blurry != 50.0:
        blur_direction = (blurry - 50.0) / 50.0
        targets["sharpness"] = float(np.clip(targets["sharpness"] - blur_direction * 0.38, 0.02, 0.98))
    return targets


def _feature_distance(features: FeatureMap, targets: FeatureMap) -> float:
    weights = {
        "brightness": 0.75,
        "contrast": 1.05,
        "saturation": 0.95,
        "warmth": 0.80,
        "sharpness": 1.10,
        "focus": 0.95,
        "entropy": 0.55,
    }
    total = 0.0
    for key, weight in weights.items():
        total += weight * abs(features[key] - targets[key])
    return total / sum(weights.values())


def _pixel_drift(image: Image.Image, base: Image.Image) -> float:
    arr = _as_array(image)
    base_arr = _as_array(base.resize(image.size, Image.Resampling.LANCZOS))
    return float(np.sqrt(np.mean((arr - base_arr) ** 2)))


def _noise_penalty(image: Image.Image) -> float:
    arr = _as_array(image)
    lum = _luminance(arr)
    gradient = _gradient_strength(lum)
    clipped = float(((arr < 0.015) | (arr > 0.985)).mean())
    high_freq = float(np.percentile(gradient, 95))
    return clipped * 0.45 + max(0.0, high_freq - 0.22) * 0.28


def score_image(image: Image.Image, base: Image.Image, targets: FeatureMap, drift_budget: float) -> Dict[str, float]:
    features = image_features(image)
    drift = _pixel_drift(image, base)
    budget = float(np.clip(drift_budget, 0.08, 0.55))
    feature_distance = _feature_distance(features, targets)
    perception_reward = 1.0 - feature_distance
    plausibility = -((drift / budget) ** 2) - _noise_penalty(image)
    energy = perception_reward + plausibility * 0.42
    result = {
        "energy": float(energy),
        "perception_reward": float(perception_reward),
        "plausibility": float(plausibility),
        "drift": float(drift),
        "feature_distance": float(feature_distance),
    }
    for key, value in features.items():
        result["feature_%s" % key] = float(value)
    return result


def score_image_with_perception(
    image: Image.Image,
    base: Image.Image,
    perception: PerceptionMap,
    targets: FeatureMap,
    drift_budget: float,
) -> Dict[str, float]:
    result = score_image(image=image, base=base, targets=targets, drift_budget=drift_budget)
    if evaluator_available():
        embedding = evaluator_energy(image, perception)
        result.update(embedding)
        result["energy"] = float(
            embedding["embedding_energy"]
            + result["perception_reward"] * 0.35
            + result["plausibility"] * 0.55
        )
        result["energy_source"] = 1.0
    else:
        result["embedding_energy"] = 0.0
        result["embedding_alignment"] = 0.0
        result["embedding_active_vectors"] = 0.0
        result["energy_source"] = 0.0
    return result


def _random_mask(size: Tuple[int, int], rng: np.random.Generator) -> Image.Image:
    width, height = size
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    cx = rng.uniform(-0.22, 0.22)
    cy = rng.uniform(-0.18, 0.18)
    sx = rng.uniform(0.28, 0.62)
    sy = rng.uniform(0.24, 0.58)
    angle = rng.uniform(-0.75, 0.75)
    rot_x = (xx - cx) * math.cos(angle) - (yy - cy) * math.sin(angle)
    rot_y = (xx - cx) * math.sin(angle) + (yy - cy) * math.cos(angle)
    field = np.exp(-((rot_x / sx) ** 2 + (rot_y / sy) ** 2))
    field = np.clip((field - 0.12) / 0.88, 0.0, 1.0)
    alpha = (field * rng.uniform(120, 220)).astype(np.uint8)
    return Image.fromarray(alpha, mode="L").filter(ImageFilter.GaussianBlur(radius=rng.uniform(12.0, 30.0)))


def _apply_warmth(image: Image.Image, direction: float, amount: float) -> Image.Image:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    arr[..., 0] *= 1.0 + direction * amount * 0.17
    arr[..., 1] *= 1.0 + direction * amount * 0.035
    arr[..., 2] *= 1.0 - direction * amount * 0.12
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


def _apply_focus(image: Image.Image, direction: float, amount: float, rng: np.random.Generator) -> Image.Image:
    if abs(direction) < 0.04:
        return image
    blurred = image.filter(ImageFilter.GaussianBlur(radius=1.6 + amount * 2.4))
    sharpened = image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=int(120 + amount * 150), threshold=3))
    mask = _random_mask(image.size, rng)
    if direction > 0:
        background = blurred
        subject = sharpened
    else:
        background = sharpened
        subject = blurred
    return Image.composite(subject, background, mask)


def propose_image(current: Image.Image, perception: PerceptionMap, rng: np.random.Generator, step_size: float) -> Image.Image:
    step = float(np.clip(step_size, 0.08, 1.0))
    proposal = current.copy()

    def direction(key: str) -> float:
        return float(np.clip((float(perception.get(key, 50.0)) - 50.0) / 50.0, -1.0, 1.0))

    proposal = ImageEnhance.Brightness(proposal).enhance(1.0 + direction("brightness") * 0.24 * step + rng.normal(0.0, 0.018))
    proposal = ImageEnhance.Contrast(proposal).enhance(1.0 + direction("contrast") * 0.34 * step + rng.normal(0.0, 0.02))
    proposal = ImageEnhance.Color(proposal).enhance(1.0 + direction("saturation") * 0.34 * step + rng.normal(0.0, 0.018))
    proposal = ImageEnhance.Sharpness(proposal).enhance(1.0 + direction("sharpness") * 0.9 * step + rng.normal(0.0, 0.035))
    proposal = _apply_warmth(proposal, direction("warmth"), step)
    proposal = _apply_focus(proposal, direction("focus"), step, rng)

    if rng.random() < 0.42:
        proposal = proposal.filter(ImageFilter.UnsharpMask(radius=0.8 + step, percent=int(70 + step * 110), threshold=4))

    arr = np.asarray(proposal.convert("RGB"), dtype=np.float32)
    arr += rng.normal(0.0, 2.0 + step * 3.5, size=arr.shape)
    proposal = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")

    mask = _random_mask(current.size, rng)
    mixed = Image.composite(proposal, current, mask)
    return ImageChops.blend(current, mixed, alpha=float(np.clip(0.42 + step * 0.38, 0.0, 1.0)))


@dataclass
class ChainState:
    prompt: str
    base: Image.Image
    current: Image.Image
    seed: int
    chain_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    iteration: int = 0
    history: List[Dict[str, float]] = field(default_factory=list)

    def rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed + self.iteration * 9973)

    def current_metrics(self, perception: PerceptionMap, drift_budget: float) -> Dict[str, float]:
        targets = perception_to_targets(perception, image_features(self.base))
        return score_image_with_perception(self.current, self.base, perception, targets, drift_budget)


class MetropolisImageOptimizer:
    def __init__(self) -> None:
        self._chains: Dict[str, ChainState] = {}

    def create_chain(self, prompt: str, image: Image.Image, seed: int) -> ChainState:
        chain = ChainState(prompt=prompt, base=image.copy(), current=image.copy(), seed=seed)
        self._chains[chain.chain_id] = chain
        return chain

    def get_chain(self, chain_id: str) -> ChainState:
        if chain_id not in self._chains:
            raise KeyError(chain_id)
        return self._chains[chain_id]

    def step(
        self,
        chain_id: str,
        perception: PerceptionMap,
        temperature: float,
        drift_budget: float,
        step_size: float,
    ) -> Dict[str, object]:
        chain = self.get_chain(chain_id)
        rng = chain.rng()
        targets = perception_to_targets(perception, image_features(chain.base))
        current_score = score_image_with_perception(chain.current, chain.base, perception, targets, drift_budget)
        proposal = propose_image(chain.current, perception, rng, step_size)
        proposal_score = score_image_with_perception(proposal, chain.base, perception, targets, drift_budget)

        temp = float(np.clip(temperature, 0.04, 2.5))
        delta = float(proposal_score["energy"] - current_score["energy"])
        acceptance_probability = float(min(1.0, math.exp(np.clip(delta / temp, -60.0, 0.0))))
        accepted = bool(rng.random() < acceptance_probability)

        if accepted:
            chain.current = proposal
            selected_score = proposal_score
        else:
            selected_score = current_score

        chain.iteration += 1
        row = {
            "iteration": float(chain.iteration),
            "accepted": 1.0 if accepted else 0.0,
            "acceptance_probability": acceptance_probability,
            "energy": float(selected_score["energy"]),
            "proposal_energy": float(proposal_score["energy"]),
            "drift": float(selected_score["drift"]),
            "perception_reward": float(selected_score["perception_reward"]),
            "plausibility": float(selected_score["plausibility"]),
        }
        chain.history.append(row)
        chain.history = chain.history[-64:]

        return {
            "chain": chain,
            "proposal": proposal,
            "accepted": accepted,
            "acceptance_probability": acceptance_probability,
            "current_score": selected_score,
            "proposal_score": proposal_score,
            "history": chain.history,
        }

    def refine(
        self,
        chain_id: str,
        perception: PerceptionMap,
        temperature: float,
        drift_budget: float,
        step_size: float,
        style: str = "auto",
    ) -> Dict[str, object]:
        chain = self.get_chain(chain_id)
        targets = perception_to_targets(perception, image_features(chain.base))
        current_score = score_image_with_perception(chain.current, chain.base, perception, targets, drift_budget)
        strength = float(np.clip(0.20 + step_size * 0.20, 0.25, 0.40))
        candidate_count = max(1, min(int(os.getenv("IMAGE_REFINE_CANDIDATES", "3")), 8))
        proposal = None
        proposal_score = None
        for candidate_index in range(candidate_count):
            candidate = refine_image(
                image=chain.current,
                prompt=chain.prompt,
                perception=perception,
                seed=chain.seed + chain.iteration * 104729 + candidate_index * 8191,
                strength=strength,
                style=style,
            )
            candidate_score = score_image_with_perception(candidate, chain.base, perception, targets, drift_budget)
            if proposal_score is None or candidate_score["energy"] > proposal_score["energy"]:
                proposal = candidate
                proposal_score = candidate_score
        assert proposal is not None
        assert proposal_score is not None

        temp = float(np.clip(temperature, 0.04, 2.5))
        delta = float(proposal_score["energy"] - current_score["energy"])
        acceptance_probability = float(min(1.0, math.exp(np.clip(delta / temp, -60.0, 0.0))))
        accepted = bool(chain.rng().random() < acceptance_probability)

        if accepted:
            chain.current = proposal
            selected_score = proposal_score
        else:
            selected_score = current_score

        chain.iteration += 1
        row = {
            "iteration": float(chain.iteration),
            "accepted": 1.0 if accepted else 0.0,
            "acceptance_probability": acceptance_probability,
            "energy": float(selected_score["energy"]),
            "proposal_energy": float(proposal_score["energy"]),
            "drift": float(selected_score["drift"]),
            "perception_reward": float(selected_score["perception_reward"]),
            "plausibility": float(selected_score["plausibility"]),
            "mode": 2.0,
        }
        chain.history.append(row)
        chain.history = chain.history[-64:]

        return {
            "chain": chain,
            "proposal": proposal,
            "accepted": accepted,
            "acceptance_probability": acceptance_probability,
            "current_score": selected_score,
            "proposal_score": proposal_score,
            "history": chain.history,
        }


optimizer = MetropolisImageOptimizer()
