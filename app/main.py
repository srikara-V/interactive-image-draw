from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.services.generator import available_styles, generate_image
from app.services.image_io import image_to_data_url, read_upload_image
from app.services.optimizer import image_features, optimizer


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=500)
    seed: int = 7
    width: int = 768
    height: int = 768
    style: str = "auto"


class StepRequest(BaseModel):
    chain_id: str
    perception: Dict[str, float]
    temperature: float = 0.38
    drift_budget: float = 0.22
    step_size: float = 0.42
    steps: int = 1


class ResetRequest(BaseModel):
    chain_id: str


app = FastAPI(
    title="Latent Atelier API",
    description="Prompt-conditioned image generation plus MH-guided perceptual image optimization.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _chain_payload(chain, proposal=None, accepted: Optional[bool] = None, acceptance_probability: Optional[float] = None, metrics=None):
    current_metrics = metrics or {
        "feature_%s" % key: value
        for key, value in image_features(chain.current).items()
    }
    payload = {
        "chain_id": chain.chain_id,
        "prompt": chain.prompt,
        "iteration": chain.iteration,
        "current": image_to_data_url(chain.current),
        "base": image_to_data_url(chain.base),
        "metrics": current_metrics,
        "history": chain.history,
    }
    if proposal is not None:
        payload["proposal"] = image_to_data_url(proposal)
    if accepted is not None:
        payload["accepted"] = accepted
    if acceptance_probability is not None:
        payload["acceptance_probability"] = acceptance_probability
    return payload


@app.get("/api/health")
def health() -> Dict[str, object]:
    return {"ok": True, "styles": available_styles()}


@app.post("/api/generate")
def generate(request: GenerateRequest) -> Dict[str, object]:
    image = generate_image(
        prompt=request.prompt,
        seed=request.seed,
        width=request.width,
        height=request.height,
        style=request.style,
    )
    chain = optimizer.create_chain(prompt=request.prompt, image=image, seed=request.seed)
    return _chain_payload(chain)


@app.post("/api/invert")
async def invert_upload(
    file: UploadFile = File(...),
    prompt: str = Form("uploaded image"),
    seed: int = Form(7),
) -> Dict[str, object]:
    raw = await file.read()
    try:
        image = read_upload_image(raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not read uploaded image") from exc
    chain = optimizer.create_chain(prompt=prompt, image=image, seed=seed)
    return _chain_payload(chain)


@app.post("/api/step")
def optimize_step(request: StepRequest) -> Dict[str, object]:
    try:
        last_result = None
        total_steps = max(1, min(int(request.steps), 24))
        for _ in range(total_steps):
            last_result = optimizer.step(
                chain_id=request.chain_id,
                perception=request.perception,
                temperature=request.temperature,
                drift_budget=request.drift_budget,
                step_size=request.step_size,
            )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown chain_id") from exc

    assert last_result is not None
    return _chain_payload(
        last_result["chain"],
        proposal=last_result["proposal"],
        accepted=last_result["accepted"],
        acceptance_probability=last_result["acceptance_probability"],
        metrics=last_result["current_score"],
    )


@app.post("/api/reset")
def reset_chain(request: ResetRequest) -> Dict[str, object]:
    try:
        chain = optimizer.get_chain(request.chain_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown chain_id") from exc
    chain.current = chain.base.copy()
    chain.iteration = 0
    chain.history = []
    return _chain_payload(chain)
