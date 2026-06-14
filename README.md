# StoryDraw

StoryDraw turns rough doodles into cartoon storybook images with a React
frontend and a Modal GPU backend.

The repo is intentionally split for deployment:

- `apps/web`: Vercel-ready React app
- `services/modal`: Modal API service for diffusion inference
- `semantic-draw`: upstream SemanticDraw research code retained as reference

## Local Frontend

```bash
cd apps/web
npm install
VITE_MODAL_API_URL="https://srikarv05--story-cartoon-api-api.modal.run" npm run dev
```

## Modal API

Use your private Modal profile:

```bash
python3 -m modal profile activate srikarv05
python3 -m modal serve services/modal/story_cartoon_api.py
```

For a stable backend URL:

```bash
python3 -m modal deploy services/modal/story_cartoon_api.py
```

Then set `VITE_MODAL_API_URL` in Vercel to the deployed Modal URL.

## Vercel

Root deploy works through `vercel.json`.

Environment variable:

```text
VITE_MODAL_API_URL=<your Modal API URL>
```

## Current Model

- Base: `runwayml/stable-diffusion-v1-5`
- Control: `lllyasviel/sd-controlnet-scribble`
- Speed: `latent-consistency/lcm-lora-sdv1-5`
- GPU: Modal A10G
