# Perception Vector Generation

This folder contains the model-specific semantic directions used by the `Refine` button.

The generator extracts text-encoder embedding deltas from the same Hugging Face Diffusers model configured for the app. Each vector is defined by a positive phrase and a negative phrase:

```text
direction = encode(positive_phrase) - encode(negative_phrase)
```

The runtime uses the saved metadata plus the phrase pairs to build weighted img2img refinement prompts. The JSON also stores normalized embedding vectors so the project has an auditable artifact showing that the directions came from the active diffusion model, not hand-wavy slider labels.

Run:

```bash
source ../.venv/bin/activate
python generate_vectors.py
```
