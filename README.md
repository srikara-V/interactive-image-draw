# Latent Atelier

Latent Atelier is a web image-editing prototype built around a Metropolis-Hastings sampler for perceptual edits. The production-grade idea is DDIM inversion plus latent steering: invert an image into a diffusion latent, propose a step along an edit/perception vector, then accept or reject it with an MH rule that balances the edit objective against the base model's plausibility distribution.

This repo ships a fast local version of that workflow so the app runs without a GPU or paid API key. The backend generates a prompt-conditioned image, treats image features as a lightweight latent state, proposes localized latent-style edits, and accepts them with the same objective/plausibility tradeoff a diffusion implementation would use.

## Why This Project Is Interesting

- Real interactive system, not a static demo.
- Clear algorithmic spine: proposal distribution, objective function, plausibility prior, MH acceptance, chain history.
- Recruiter-friendly architecture: typed frontend, tested backend, documented path from CPU prototype to diffusion latent backend.
- Runs locally in minutes while leaving a credible route to Stable Diffusion/DDIM integration.

## Stack

- Backend: FastAPI, Pillow, NumPy
- Frontend: React, TypeScript, Vite, lucide-react
- Tests: pytest

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd frontend
npm install
cd ..
```

## Run

Start the API:

```bash
source .venv/bin/activate
uvicorn app.main:app --reload
```

Start the web app in another terminal:

```bash
cd frontend
npm run dev
```

Open the Vite URL, usually `http://localhost:5173`.

## How The Sampler Works

For a current image state `x`, the optimizer proposes a new image `x'` by applying a localized edit guided by the user's perception vector:

```text
q(x' | x, v) = localized_steer(x, perception_vector=v, noise=epsilon)
```

The proposal is scored by:

```text
energy(x) = perception_reward(x, target) + plausibility_log_prior(x | x_base)
```

The proposal is accepted with:

```text
alpha = min(1, exp((energy(x') - energy(x)) / temperature))
```

The local implementation uses contrast, brightness, warmth, saturation, sharpness, focus, entropy, and base-image drift as surrogate features. A diffusion version would replace the surrogate latent with DDIM-inverted latents and replace the plausibility score with model log-probability or a calibrated diffusion prior.

See [docs/algorithm.md](docs/algorithm.md) for the deeper implementation plan.

## GitHub Remote

The intended remote is:

```bash
git remote add origin https://github.com/srikara-V/image-generation-and-optimization.git
```
