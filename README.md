# StoryDraw

StoryDraw is a sketch-conditioned cartoon generation app built around the
SemanticDraw idea of interactive visual creation. A user draws a rough scene,
keeps an optional prompt in plain language, and sends the doodle to a GPU-backed
diffusion service that turns the sketch into a cleaner children's storybook
cartoon.

The project is structured as a deployable product rather than a notebook demo.
The frontend is a Vercel-ready React app, while model inference lives behind a
Modal API service so GPU work is isolated from the browser experience.

## What Is In This Repo

- `apps/web` contains the React drawing interface, prompt bar, live generation
  flow, and warm/cool controls for the remote model container.
- `services/modal` contains the Modal API service that loads Stable Diffusion
  1.5 with ControlNet Scribble conditioning and an LCM LoRA for lower-latency
  inference.
- `semantic-draw` keeps the upstream SemanticDraw research code available as a
  reference point for interactive drawing workflows.
- `docs` contains architecture notes for the frontend/backend split and model
  serving path.

## Product Shape

The app is designed around a side-by-side creative loop: the left panel is the
input doodle and the right panel is the generated cartoon output. Drawing is
locked until the Modal container is warmed, which makes GPU state explicit in
the UI and avoids silent paid inference startup. The prompt is optional; when it
is left empty, the app uses a default prompt tuned for simple, layout-preserving
storybook cartoon output.

## Model Path

Generation is handled through a real diffusion pipeline on Modal:

- Base image model: Stable Diffusion 1.5
- Sketch conditioning: ControlNet Scribble
- Latency optimization: LCM LoRA
- GPU target: Modal A10G

The frontend sends the current canvas image and prompt to the Modal API. The
backend preprocesses the doodle into a scribble control image, runs diffusion
inference, and returns the generated image as a browser-renderable response.
