import base64
import gc
import os
from io import BytesIO

import modal


app = modal.App("story-cartoon-api")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "accelerate==1.1.1",
        "diffusers==0.31.0",
        "fastapi[standard]==0.115.6",
        "huggingface-hub==0.26.5",
        "peft==0.13.2",
        "pillow==10.4.0",
        "safetensors==0.4.5",
        "torch==2.5.1",
        "transformers==4.46.3",
    )
)

MODEL_ID = "runwayml/stable-diffusion-v1-5"
CONTROLNET_ID = "lllyasviel/sd-controlnet-scribble"
LCM_LORA_ID = "latent-consistency/lcm-lora-sdv1-5"

_pipe = None
_device = None


def _decode_data_url(data_url: str):
    from PIL import Image, ImageOps

    _, encoded = data_url.split(",", 1)
    image = Image.open(BytesIO(base64.b64decode(encoded))).convert("RGBA")
    white = Image.new("RGBA", image.size, "white")
    white.alpha_composite(image)
    image = white.convert("RGB").resize((512, 512))
    gray = ImageOps.grayscale(image)
    return gray.point(lambda p: 255 if p < 245 else 0).convert("RGB")


def _encode_image(image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")


def _load_pipeline():
    global _pipe, _device
    if _pipe is not None:
        return _pipe

    import torch
    from diffusers import ControlNetModel, LCMScheduler, StableDiffusionControlNetPipeline

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if _device == "cuda" else torch.float32

    controlnet = ControlNetModel.from_pretrained(CONTROLNET_ID, torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        MODEL_ID,
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=dtype,
    )
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(LCM_LORA_ID)
    pipe.fuse_lora()
    pipe.to(_device)
    pipe.enable_attention_slicing()
    _pipe = pipe
    return _pipe


def _unload_pipeline() -> None:
    global _pipe
    _pipe = None
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


@app.function(
    image=image,
    gpu="A10G",
    timeout=900,
    scaledown_window=60,
)
@modal.asgi_app()
def api():
    import time

    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field

    web = FastAPI(title="Story Cartoon API")

    allowed_origins = [
        origin.strip()
        for origin in os.environ.get("ALLOWED_ORIGINS", "*").split(",")
        if origin.strip()
    ]
    web.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    class GenerateRequest(BaseModel):
        image: str = Field(..., description="PNG data URL from the doodle canvas")
        prompt: str = Field(..., min_length=1)

    @web.get("/")
    async def root():
        return {"service": "story-cartoon-api", "model_loaded": _pipe is not None}

    @web.get("/health")
    async def health():
        return {"ok": True, "model_loaded": _pipe is not None}

    @web.post("/warmup")
    async def warmup():
        started = time.monotonic()
        _load_pipeline()
        return {"ok": True, "model_loaded": True, "elapsed_seconds": time.monotonic() - started}

    @web.post("/cooldown")
    async def cooldown():
        _unload_pipeline()
        return {"ok": True, "model_loaded": False}

    @web.post("/generate")
    async def generate(req: GenerateRequest):
        if _pipe is None:
            raise HTTPException(status_code=409, detail="Model is not warm. Press Warm up first.")

        import torch

        started = time.monotonic()
        guidance = _decode_data_url(req.image)
        generator = torch.Generator(device=_device).manual_seed(2026) if _device == "cuda" else None
        result = _pipe(
            prompt=req.prompt,
            image=guidance,
            negative_prompt=(
                "photorealistic, realistic photo, extra characters, extra faces, extra objects, "
                "crowd, busy background, complex scenery, random animals, random flowers, "
                "unrelated objects, changed composition, warped pose, unreadable clutter, "
                "ugly, blurry, messy, distorted, low quality, watermark"
            ),
            num_inference_steps=8,
            guidance_scale=1.8,
            controlnet_conditioning_scale=1.45,
            control_guidance_start=0.0,
            control_guidance_end=1.0,
            generator=generator,
            width=512,
            height=512,
        ).images[0]
        return {
            "image": _encode_image(result),
            "elapsed_seconds": time.monotonic() - started,
            "model": {
                "base": MODEL_ID,
                "controlnet": CONTROLNET_ID,
                "lora": LCM_LORA_ID,
            },
        }

    return web
