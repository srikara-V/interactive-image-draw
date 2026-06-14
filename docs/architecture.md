# Architecture

StoryDraw is split into a browser frontend and a Modal GPU API.

## Frontend

- `apps/web`
- React + Vite
- Designed for Vercel static hosting
- Talks to Modal through `VITE_MODAL_API_URL`

## Backend

- `services/modal/story_cartoon_api.py`
- Modal ASGI app
- A10G GPU
- Explicit `Warm up` and `Cool down` endpoints
- Diffusion stack:
  - `runwayml/stable-diffusion-v1-5`
  - `lllyasviel/sd-controlnet-scribble`
  - `latent-consistency/lcm-lora-sdv1-5`

## Request Flow

1. User draws on the React canvas.
2. Frontend sends a PNG data URL and story prompt to Modal.
3. Modal converts the doodle to ControlNet scribble guidance.
4. Diffusion returns a generated PNG data URL.
5. Frontend renders the image side by side with the doodle.

## Lifecycle

`POST /warmup` loads the model into the GPU container.

`POST /cooldown` unloads the model and clears CUDA memory. Modal scales the
container down after `scaledown_window=60`, meaning roughly 60 seconds of idle
time before shutdown.
