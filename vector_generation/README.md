# Perception Vector Generation

This folder contains the model-specific semantic directions used by the `Refine` button.

The generator extracts text-encoder embedding deltas from the same Hugging Face Diffusers model configured for the app. Each vector is defined by a positive phrase and a negative phrase:

```text
direction = encode(positive_phrase) - encode(negative_phrase)
```

The runtime loads `perception_vectors.pt` and adds the weighted tensor direction directly to the diffusion conditioning embeddings before img2img denoising. `perception_vectors.json` is kept only as a readable audit artifact.

Run:

```bash
source ../.venv/bin/activate
python generate_vectors.py
```
