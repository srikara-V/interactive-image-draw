import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
import torch
from diffusers import DDIMInverseScheduler, DDIMScheduler
from PIL import Image, ImageEnhance, ImageFilter

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


def _image_to_tensor(image: Image.Image, device: str, dtype: torch.dtype) -> torch.Tensor:
    image = image.convert("RGB")
    width, height = image.size
    width = width - (width % 8)
    height = height - (height % 8)
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 255.0
    values = torch.from_numpy(array).permute(2, 0, 1)
    values = values.unsqueeze(0) * 2.0 - 1.0
    return values.to(device=device, dtype=dtype)


def _latents_to_image(pipeline, latents: torch.Tensor) -> Image.Image:
    scaling = pipeline.vae.config.scaling_factor
    decoded = pipeline.vae.decode(latents / scaling, return_dict=False)[0]
    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    array = decoded.detach().cpu().permute(0, 2, 3, 1).float().numpy()[0]
    return Image.fromarray((array * 255).round().astype("uint8"), mode="RGB")


def _encode_image_latents(pipeline, image: Image.Image, settings: DiffusionSettings) -> torch.Tensor:
    tensor = _image_to_tensor(image, settings.device, settings.dtype)
    with torch.no_grad():
        latents = pipeline.vae.encode(tensor).latent_dist.mode()
    return latents * pipeline.vae.config.scaling_factor


def _noise_prediction(
    pipeline,
    scheduler,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    do_guidance = guidance_scale > 1.0 and negative_prompt_embeds is not None
    if do_guidance:
        latent_model_input = torch.cat([latents, latents])
        encoder_hidden_states = torch.cat([negative_prompt_embeds, prompt_embeds])
    else:
        latent_model_input = latents
        encoder_hidden_states = prompt_embeds
    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)
    noise_pred = pipeline.unet(
        latent_model_input,
        timestep,
        encoder_hidden_states=encoder_hidden_states,
        return_dict=False,
    )[0]
    if do_guidance:
        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)
    return noise_pred


def _ddim_invert(
    pipeline,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    guidance_scale: float,
    steps: int,
) -> torch.Tensor:
    inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
    inverse_scheduler.set_timesteps(steps, device=latents.device)
    inverted = latents
    with torch.no_grad():
        for timestep in inverse_scheduler.timesteps:
            noise_pred = _noise_prediction(
                pipeline,
                inverse_scheduler,
                inverted,
                timestep,
                prompt_embeds,
                negative_prompt_embeds,
                guidance_scale,
            )
            inverted = inverse_scheduler.step(noise_pred, timestep, inverted, return_dict=False)[0]
    return inverted


def _ddim_sample(
    pipeline,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    guidance_scale: float,
    steps: int,
) -> torch.Tensor:
    scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    scheduler.set_timesteps(steps, device=latents.device)
    sampled = latents
    with torch.no_grad():
        for timestep in scheduler.timesteps:
            noise_pred = _noise_prediction(
                pipeline,
                scheduler,
                sampled,
                timestep,
                prompt_embeds,
                negative_prompt_embeds,
                guidance_scale,
            )
            sampled = scheduler.step(noise_pred, timestep, sampled, return_dict=False)[0]
    return sampled


def _latent_feature_gradient(pipeline, latents: torch.Tensor, perception: Dict[str, float]) -> torch.Tensor:
    def weight(key: str) -> float:
        return max(0.0, min(float(perception.get(key, 0.0)) / 100.0, 1.0))

    contrast_weight = weight("contrast")
    saturation_weight = weight("saturation")
    warmth_weight = weight("warmth")
    blurry_weight = weight("blurry")
    sharpness_weight = weight("sharpness")
    if blurry_weight >= sharpness_weight:
        sharpness_weight = -blurry_weight
    else:
        sharpness_weight = sharpness_weight
    if max(abs(contrast_weight), abs(saturation_weight), abs(warmth_weight), abs(sharpness_weight)) < 0.04:
        return torch.zeros_like(latents)

    working = latents.detach().clone().requires_grad_(True)
    decoded = pipeline.vae.decode(working / pipeline.vae.config.scaling_factor, return_dict=False)[0]
    image = (decoded / 2 + 0.5).clamp(0, 1)
    red, green, blue = image[:, 0], image[:, 1], image[:, 2]
    lum = red * 0.2126 + green * 0.7152 + blue * 0.0722

    contrast = lum.std()
    max_channel = image.max(dim=1).values
    min_channel = image.min(dim=1).values
    saturation = ((max_channel - min_channel) / max_channel.clamp(min=1e-4)).mean()
    warmth = (red.mean() - blue.mean()) * 0.9
    grad_x = (lum[:, :, 1:] - lum[:, :, :-1]).abs().mean()
    grad_y = (lum[:, 1:, :] - lum[:, :-1, :]).abs().mean()
    sharpness = grad_x + grad_y

    objective = (
        contrast_weight * contrast
        + saturation_weight * saturation
        + warmth_weight * warmth
        + sharpness_weight * sharpness
    )
    objective.backward()
    gradient = working.grad.detach()
    norm = torch.linalg.vector_norm(gradient)
    if not torch.isfinite(norm) or float(norm.item()) < 1e-8:
        return torch.zeros_like(latents)
    return gradient / norm


def _apply_warmth_image(image: Image.Image, amount: float) -> Image.Image:
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    array[..., 0] *= 1.0 + amount * 0.22
    array[..., 1] *= 1.0 + amount * 0.04
    array[..., 2] *= 1.0 - amount * 0.16
    return Image.fromarray(np.clip(array, 0, 255).astype("uint8"), mode="RGB")


def _perception_target_image(image: Image.Image, perception: Dict[str, float]) -> Image.Image:
    def weight(key: str) -> float:
        return max(0.0, min(float(perception.get(key, 0.0)) / 100.0, 1.0))

    target = image.convert("RGB")
    blurry = weight("blurry")
    sharpness = weight("sharpness")
    contrast = weight("contrast")
    saturation = weight("saturation")
    warmth = weight("warmth")

    blur_amount = blurry if blurry >= sharpness else 0.0
    sharp_amount = sharpness if sharpness > blurry else 0.0
    if blur_amount > 0.04:
        target = target.filter(ImageFilter.GaussianBlur(radius=2.0 + blur_amount * 10.0))
    if sharp_amount > 0.04:
        target = target.filter(
            ImageFilter.UnsharpMask(
                radius=0.8 + sharp_amount * 0.8,
                percent=int(120 + sharp_amount * 220),
                threshold=1,
            )
        )
    if abs(contrast) > 0.04:
        target = ImageEnhance.Contrast(target).enhance(1.0 + contrast * 0.75)
    if abs(saturation) > 0.04:
        target = ImageEnhance.Color(target).enhance(1.0 + saturation * 0.95)
    if abs(warmth) > 0.04:
        target = _apply_warmth_image(target, warmth)
    return target


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
    guidance_scale = max(_img2img_guidance_scale(), 1.05)
    prompt_embeds, negative_prompt_embeds = _encode_prompt_embeddings(
        pipeline=pipeline,
        prompt=base_prompt,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        guidance_scale=guidance_scale,
        settings=settings,
    )

    size = image.size
    steps = _img2img_steps()
    base_latents = _encode_image_latents(pipeline, image, settings)
    inverted = _ddim_invert(
        pipeline=pipeline,
        latents=base_latents,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        guidance_scale=guidance_scale,
        steps=steps,
    )
    target_image = _perception_target_image(image, perception)
    target_latents = _encode_image_latents(pipeline, target_image, settings)
    target_inverted = _ddim_invert(
        pipeline=pipeline,
        latents=target_latents,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        guidance_scale=guidance_scale,
        steps=steps,
    )
    target_direction = target_inverted - inverted
    direction = steering_vector(
        perception,
        device=settings.device,
        dtype=inverted.dtype,
        latent_shape=inverted.shape,
    )
    gradient_direction = _latent_feature_gradient(pipeline, base_latents, perception).to(
        device=settings.device,
        dtype=inverted.dtype,
    )
    scale = float(os.getenv("IMAGE_LATENT_STEERING_SCALE", "5.5"))
    gradient_scale = float(os.getenv("IMAGE_GRADIENT_STEERING_SCALE", "8.0"))
    target_scale = float(os.getenv("IMAGE_TARGET_LATENT_STEERING_SCALE", "0.85"))
    proposal_direction = direction * scale + gradient_direction * gradient_scale + target_direction * target_scale
    noise_scale = float(os.getenv("IMAGE_REFINE_NOISE_SCALE", "0.11"))
    noise = torch.randn(
        inverted.detach().cpu().shape,
        generator=_generator_for_seed(seed, "cpu"),
        dtype=torch.float32,
    ).to(device=settings.device, dtype=inverted.dtype)
    proposal_latents = inverted + proposal_direction * float(max(0.05, min(strength, 0.75))) + noise * noise_scale
    sampled = _ddim_sample(
        pipeline=pipeline,
        latents=proposal_latents,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        guidance_scale=guidance_scale,
        steps=steps,
    )
    return _latents_to_image(pipeline, sampled).resize(size, Image.Resampling.LANCZOS)


def model_info() -> Dict[str, str]:
    settings = _settings()
    return {
        "model_id": settings.model_id,
        "device": settings.device,
        "dtype": str(settings.dtype).replace("torch.", ""),
    }


def available_styles() -> List[str]:
    return ["auto"] + sorted(STYLE_PROMPTS.keys())
