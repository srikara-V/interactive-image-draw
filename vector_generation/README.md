# Perception Vector Generation

This folder contains the model-specific semantic directions used by the `Refine` button.

The generator creates paired edits of the same generated images, DDIM-inverts each edited pair, and averages the latent deltas:

```text
direction = DDIMInvert(edited_positive_image) - DDIMInvert(edited_negative_image)
```

The runtime loads `perception_vectors.pt` and adds the weighted latent direction after DDIM inversion. `perception_vectors.json` is kept only as a readable audit artifact.

Run:

```bash
source ../.venv/bin/activate
python generate_vectors.py
```

The MH acceptance energy uses a second set of image-space evaluator vectors that follow the reference project pattern:

```text
direction = mean(CLIP(positive_images)) - mean(CLIP(negative_images))
energy = CLIP(candidate_image) · direction
```

Generate them with:

```bash
source ../.venv/bin/activate
python generate_evaluator_vectors.py
```

This writes `evaluator_vectors.pt` for runtime scoring, `evaluator_vectors.json` for audit, and `evaluator_examples/` so the positive/negative examples can be inspected.
