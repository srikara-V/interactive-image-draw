# Algorithm Notes

## Goal

Build a web image editor that feels like an optimization lab: instead of applying deterministic filters, the user defines a perceptual target and the system samples plausible edits. The important constraint is that edits should improve the requested direction without drifting into obviously broken or unnatural images.

## Diffusion Version

The high-quality version uses a diffusion model:

1. Generate or upload an image.
2. Run DDIM inversion to recover a latent trajectory for that image.
3. Compute a perception vector from the user's edit target. This can come from CLIP, a text-image reward model, a learned aesthetic model, or a hand-authored direction in latent space.
4. Propose a latent edit:

   ```text
   z' = z + step_size * perception_vector + sigma * noise
   ```

5. Decode or partially denoise from `z'`.
6. Score the proposal:

   ```text
   log pi(z') = task_reward(decoded(z')) + lambda * base_model_logprob(z')
   ```

7. Accept with Metropolis-Hastings:

   ```text
   alpha = min(1, exp(log pi(z') - log pi(z)))
   ```

The result is a controlled random walk through plausible images, where the base model acts as a prior and the user's edit intent acts as a reward.

## Current Implementation

The current repo uses Hugging Face Diffusers for initial text-to-image generation. The default is `stabilityai/sd-turbo`, configurable through `IMAGE_MODEL_ID`.

The sampler keeps the same MH shape but currently scores CPU-friendly image features:

- `brightness`: mean luminance
- `contrast`: luminance standard deviation
- `saturation`: mean HSV saturation
- `warmth`: relative red/blue balance
- `sharpness`: mean local gradient magnitude
- `focus`: center detail relative to border detail
- `entropy`: luminance histogram entropy
- `drift`: normalized pixel distance from the base image

The proposal distribution makes a locally masked edit, roughly equivalent to a small latent steering step:

```text
x' = mask * steer(x, target_vector, epsilon) + (1 - mask) * x
```

The local plausibility prior penalizes:

- large drift from the base image
- clipped color channels
- excessive high-frequency noise
- feature combinations far outside the generated base image

This is not pretending to be a diffusion model. It is a fast algorithmic scaffold that makes the UI, API contract, sampler diagnostics, and acceptance behavior concrete before swapping in a heavier model backend.

## Upgrade Path

The next backend adapter should implement this interface:

```python
class LatentBackend:
    def generate(prompt: str, seed: int) -> Image.Image: ...
    def invert(image: Image.Image, prompt: str) -> LatentState: ...
    def propose(latent: LatentState, vector: PerceptionVector) -> LatentState: ...
    def decode(latent: LatentState) -> Image.Image: ...
    def log_prior(latent: LatentState) -> float: ...
```

The FastAPI endpoints do not need to change. Only the internals of generation, inversion, proposal, and prior scoring need to move from the surrogate backend to the diffusion backend.
