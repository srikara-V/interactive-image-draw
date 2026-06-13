import json
import sys
from pathlib import Path
from typing import Dict

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.generator import _load_pipeline, model_info  # noqa: E402
from app.services.perception_vectors import VECTOR_DEFINITIONS  # noqa: E402


OUTPUT = Path(__file__).resolve().parent / "perception_vectors.json"


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


def generate_vectors() -> Dict[str, object]:
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
    return {
        **model_info(),
        "method": "text_encoder_mean_pool_delta",
        "vectors": vectors,
    }


def main() -> None:
    payload = generate_vectors()
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote %s" % OUTPUT)
    for name, vector in payload["vectors"].items():
        print("%s: dim=%s norm=%.4f" % (name, vector["dimension"], vector["norm"]))


if __name__ == "__main__":
    main()
