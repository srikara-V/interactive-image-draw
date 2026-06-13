import json
import sys
from pathlib import Path
from typing import Dict

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.generator import _encode_image_latents, _load_pipeline, _settings, model_info  # noqa: E402
from app.services.perception_vectors import VECTOR_DEFINITIONS  # noqa: E402


OUTPUT_JSON = Path(__file__).resolve().parent / "perception_vectors.json"
OUTPUT_PT = Path(__file__).resolve().parent / "perception_vectors.pt"


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


def _generate_latents(pipeline, phrase: str, seed: int) -> torch.Tensor:
    settings = _settings()
    generator_device = settings.device if settings.device == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    result = pipeline(
        prompt=phrase,
        negative_prompt="low quality, distorted, text, watermark",
        width=512,
        height=512,
        num_inference_steps=4,
        guidance_scale=0.0,
        generator=generator,
    )
    image = result.images[0].convert("RGB")
    return _encode_image_latents(pipeline, image, settings)[0].detach().float().cpu()


def generate_vectors() -> Dict[str, object]:
    pipeline = _load_pipeline()
    vectors = {}
    latent_vectors = {}
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
        pos_latent = _generate_latents(pipeline, positive, seed=1701)
        neg_latent = _generate_latents(pipeline, negative, seed=1701)
        latent_direction = pos_latent - neg_latent
        latent_norm = float(torch.linalg.vector_norm(latent_direction).item())
        latent_unit = latent_direction / max(latent_norm, 1e-8)
        latent_vectors[name] = {
            "shape": list(latent_unit.shape),
            "norm": latent_norm,
            "mean": float(latent_unit.mean().item()),
            "std": float(latent_unit.std().item()),
        }
    return {
        **model_info(),
        "method": "ddim_latent_delta_with_text_embedding_audit",
        "vectors": vectors,
        "latent_vectors": latent_vectors,
    }


def _tensor_payload(payload: Dict[str, object]) -> Dict[str, object]:
    vectors = payload["vectors"]
    tensor_vectors = {}
    for name, vector in vectors.items():
        tensor_vectors[name] = torch.tensor(vector["embedding"], dtype=torch.float32)
    latent_tensors = {}
    pipeline = _load_pipeline()
    for name, definition in VECTOR_DEFINITIONS.items():
        pos_latent = _generate_latents(pipeline, definition["positive"], seed=1701)
        neg_latent = _generate_latents(pipeline, definition["negative"], seed=1701)
        direction = pos_latent - neg_latent
        latent_tensors[name] = direction / max(float(torch.linalg.vector_norm(direction).item()), 1e-8)
    return {
        "model_id": payload["model_id"],
        "device": payload["device"],
        "dtype": payload["dtype"],
        "method": payload["method"],
        "vectors": tensor_vectors,
        "latent_vectors": latent_tensors,
    }


def main() -> None:
    payload = generate_vectors()
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    torch.save(_tensor_payload(payload), OUTPUT_PT)
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
