import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional

import torch
from PIL import Image

from app.services.perception_vectors import steering_vector


DEFAULT_MODEL_ID = "stabilityai/sd-turbo"
DEFAULT_NEGATIVE_PROMPT = (
    "low quality, blurry, distorted, deformed, extra limbs, bad anatomy, text, watermark, "
    "signature, jpeg artifacts, noisy, oversaturated"
)

STYLE_PROMPTS: Dict[str, str] = {
    "cinematic": "cinematic lighting, shallow depth of field, high-end color grading, detailed composition",
    "product": "premium product photography, sharp focus, commercial studio lighting, clean composition",
    "editorial": "editorial magazine photography, refined styling, expressive lighting, polished detail",
    "concept": "high-detail concept art, dramatic atmosphere, coherent design, professional rendering",
    "abstract": "abstract generative artwork, rich texture, balanced composition, intricate details",
}


class ModelLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class DiffusionSettings:
    model_id: str
    device: str
    dtype: torch.dtype


def _pick_device() -> str:
    requested = os.getenv("IMAGE_DEVICE", "").strip().lower()
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _pick_dtype(device: str) -> torch.dtype:
    requested = os.getenv("IMAGE_DTYPE", "").strip().lower()
    if requested in {"float16", "fp16"}:
        return torch.float16
    if requested in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if requested in {"float32", "fp32"}:
        return torch.float32
    if device == "cuda":
        return torch.float16
    return torch.float32


def _settings() -> DiffusionSettings:
    device = _pick_device()
    return DiffusionSettings(
        model_id=os.getenv("IMAGE_MODEL_ID", DEFAULT_MODEL_ID).strip() or DEFAULT_MODEL_ID,
        device=device,
        dtype=_pick_dtype(device),
    )


def _style_prompt(prompt: str, style: str) -> str:
    style_suffix = STYLE_PROMPTS.get(style, "")
    quality = "best quality, high detail, coherent image, professional visual design"
    if style_suffix:
        return "%s, %s, %s" % (prompt, style_suffix, quality)
    return "%s, %s" % (prompt, quality)


@lru_cache(maxsize=1)
def _load_pipeline():
    settings = _settings()
    try:
        from diffusers import AutoPipelineForText2Image
    except ImportError as exc:
        raise ModelLoadError(
            "Diffusers is not installed. Run `pip install -r requirements.txt` in the virtual environment."
        ) from exc

    try:
        kwargs = {
            "torch_dtype": settings.dtype,
            "use_safetensors": True,
        }
        if settings.dtype is torch.float16:
            kwargs["variant"] = "fp16"
        pipeline = AutoPipelineForText2Image.from_pretrained(settings.model_id, **kwargs)
    except Exception as exc:
        raise ModelLoadError(
            "Could not load Hugging Face diffusion model `%s`. Check network access, disk space, and model permissions."
            % settings.model_id
        ) from exc

    pipeline = pipeline.to(settings.device)
    if hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing()
    if settings.device == "cuda" and hasattr(pipeline, "enable_model_cpu_offload"):
        try:
            pipeline.enable_model_cpu_offload()
        except Exception:
            pass
    return pipeline


@lru_cache(maxsize=1)
def _load_img2img_pipeline():
    settings = _settings()
    try:
        from diffusers import AutoPipelineForImage2Image
    except ImportError as exc:
        raise ModelLoadError(
            "Diffusers is not installed. Run `pip install -r requirements.txt` in the virtual environment."
        ) from exc

    try:
        pipeline = AutoPipelineForImage2Image.from_pipe(_load_pipeline())
    except Exception:
        try:
            kwargs = {
                "torch_dtype": settings.dtype,
                "use_safetensors": True,
            }
            if settings.dtype is torch.float16:
                kwargs["variant"] = "fp16"
            pipeline = AutoPipelineForImage2Image.from_pretrained(settings.model_id, **kwargs)
        except Exception as exc:
            raise ModelLoadError(
                "Could not load Hugging Face img2img pipeline for `%s`." % settings.model_id
            ) from exc

    pipeline = pipeline.to(settings.device)
    if hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing()
    return pipeline


def _generator_for_seed(seed: int, device: str) -> torch.Generator:
    generator_device = device if device == "cuda" else "cpu"
    return torch.Generator(device=generator_device).manual_seed(int(seed))


def _recommended_steps() -> int:
    raw = os.getenv("IMAGE_NUM_INFERENCE_STEPS", "").strip()
    if raw:
        return max(1, min(int(raw), 80))
    model_id = _settings().model_id.lower()
    if "turbo" in model_id or "lightning" in model_id:
        return 4
    return 28


def _guidance_scale() -> float:
    raw = os.getenv("IMAGE_GUIDANCE_SCALE", "").strip()
    if raw:
        return max(0.0, min(float(raw), 20.0))
    model_id = _settings().model_id.lower()
    if "turbo" in model_id or "lightning" in model_id:
        return 0.0
    return 7.0


def _img2img_guidance_scale() -> float:
    raw = os.getenv("IMAGE_REFINE_GUIDANCE_SCALE", "").strip()
    if raw:
        return max(0.0, min(float(raw), 20.0))
    model_id = _settings().model_id.lower()
    if "turbo" in model_id or "lightning" in model_id:
        return 1.2
    return 7.5


def _img2img_steps() -> int:
    raw = os.getenv("IMAGE_REFINE_STEPS", "").strip()
    if raw:
        return max(1, min(int(raw), 80))
    model_id = _settings().model_id.lower()
    if "turbo" in model_id or "lightning" in model_id:
        return 8
    return 22


def _encode_prompt_embeddings(pipeline, prompt: str, negative_prompt: str, guidance_scale: float, settings: DiffusionSettings):
    do_guidance = guidance_scale > 1.0
    encoded = pipeline.encode_prompt(
        prompt=prompt,
        device=settings.device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_guidance,
        negative_prompt=negative_prompt,
    )
    if not isinstance(encoded, tuple) or len(encoded) < 2:
        raise ModelLoadError("The diffusion pipeline did not return prompt embeddings.")
    prompt_embeds, negative_prompt_embeds = encoded[0], encoded[1]
    return prompt_embeds, negative_prompt_embeds


def _steer_prompt_embeddings(
    prompt_embeds: torch.Tensor,
    perception: Dict[str, float],
    settings: DiffusionSettings,
) -> torch.Tensor:
    direction = steering_vector(perception, device=settings.device, dtype=prompt_embeds.dtype)
    scale = float(os.getenv("IMAGE_VECTOR_STEERING_SCALE", "1.8"))
    steered = prompt_embeds + direction.view(1, 1, -1) * scale
    return steered.to(device=prompt_embeds.device, dtype=prompt_embeds.dtype)


def generate_image(prompt: str, seed: int = 7, width: int = 768, height: int = 768, style: str = "auto") -> Image.Image:
    settings = _settings()
    pipeline = _load_pipeline()
    width = int(max(512, min(width, 1024)))
    height = int(max(512, min(height, 1024)))
    width = width - (width % 8)
    height = height - (height % 8)

    full_prompt = _style_prompt(prompt, style)
    result = pipeline(
        prompt=full_prompt,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        width=width,
        height=height,
        num_inference_steps=_recommended_steps(),
        guidance_scale=_guidance_scale(),
        generator=_generator_for_seed(seed, settings.device),
    )
    image: Optional[Image.Image] = result.images[0] if result.images else None
    if image is None:
        raise ModelLoadError("The diffusion pipeline did not return an image.")
    return image.convert("RGB")


def refine_image(
    image: Image.Image,
    prompt: str,
    perception: Dict[str, float],
    seed: int,
    strength: float,
    style: str = "auto",
) -> Image.Image:
    settings = _settings()
    pipeline = _load_img2img_pipeline()
    base_prompt = _style_prompt(prompt, style)
    refine_prompt = "%s, preserve the same subject and composition" % base_prompt
    guidance_scale = _img2img_guidance_scale()
    prompt_embeds, negative_prompt_embeds = _encode_prompt_embeddings(
        pipeline=pipeline,
        prompt=refine_prompt,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        guidance_scale=guidance_scale,
        settings=settings,
    )
    prompt_embeds = _steer_prompt_embeddings(prompt_embeds, perception, settings)

    size = image.size
    init_image = image.convert("RGB")
    result = pipeline(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        image=init_image,
        strength=float(max(0.25, min(strength, 0.55))),
        num_inference_steps=_img2img_steps(),
        guidance_scale=guidance_scale,
        generator=_generator_for_seed(seed, settings.device),
    )
    refined: Optional[Image.Image] = result.images[0] if result.images else None
    if refined is None:
        raise ModelLoadError("The diffusion img2img pipeline did not return an image.")
    return refined.convert("RGB").resize(size, Image.Resampling.LANCZOS)


def model_info() -> Dict[str, str]:
    settings = _settings()
    return {
        "model_id": settings.model_id,
        "device": settings.device,
        "dtype": str(settings.dtype).replace("torch.", ""),
    }


def available_styles() -> List[str]:
    return ["auto"] + sorted(STYLE_PROMPTS.keys())
