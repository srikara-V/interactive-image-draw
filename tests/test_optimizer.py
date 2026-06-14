import app.services.generator as generator
from PIL import Image

from app.services.generator import generate_image
from app.services.optimizer import image_features, optimizer, perception_to_targets
from app.services.perception_vectors import slider_weights


class FakePipeline:
    def __call__(self, **kwargs):
        return type("Result", (), {"images": [Image.new("RGB", (kwargs["width"], kwargs["height"]), (120, 150, 190))]})()


def test_generate_image_uses_diffusion_pipeline(monkeypatch):
    fake = FakePipeline()
    calls = []

    def load_pipeline():
        calls.append(True)
        return fake

    monkeypatch.setattr(generator, "_load_pipeline", load_pipeline)
    image = generate_image("cinematic product shot", seed=11, width=512, height=512)

    assert calls
    assert image.size == (512, 512)


def test_generated_image_has_expected_features(monkeypatch):
    monkeypatch.setattr(generator, "_load_pipeline", lambda: FakePipeline())
    image = generate_image("cinematic product shot of a wearable device", seed=11, width=512, height=512)
    features = image_features(image)

    assert image.size == (512, 512)
    assert set(features) == {"brightness", "contrast", "saturation", "warmth", "sharpness", "focus", "entropy"}
    assert all(0.0 <= value <= 1.0 for value in features.values())


def test_metropolis_step_records_history(monkeypatch):
    monkeypatch.setenv("IMAGE_EVALUATOR", "features")
    monkeypatch.setattr(generator, "_load_pipeline", lambda: FakePipeline())
    image = generate_image("editorial portrait with dramatic contrast", seed=3, width=512, height=512)
    chain = optimizer.create_chain("editorial portrait with dramatic contrast", image, seed=3)
    result = optimizer.step(
        chain.chain_id,
        perception={
            "brightness": 55,
            "contrast": 80,
            "saturation": 62,
            "warmth": 58,
            "sharpness": 74,
            "focus": 80,
            "entropy": 54,
        },
        temperature=0.4,
        drift_budget=0.22,
        step_size=0.45,
    )

    assert chain.iteration == 1
    assert len(chain.history) == 1
    assert 0.0 <= result["acceptance_probability"] <= 1.0
    assert result["proposal"].size == image.size


def test_perception_vectors_map_sliders_to_model_directions():
    weights = slider_weights({"blurry": 80, "contrast": 85, "saturation": 25, "warmth": 70, "sharpness": 10})

    assert slider_weights({"sharpness": 10})["blurry"] > 0
    assert slider_weights({"blurry": 10})["sharpness"] > 0
    assert weights["blurry"] > 0
    assert weights["contrast"] > 0
    assert weights["saturation"] < 0
    assert weights["warmth"] > 0
    assert weights["sharpness"] == 0.0
    assert weights["blurry"] > 0.0


def test_blurry_slider_targets_lower_sharpness():
    base_features = {
        "brightness": 0.5,
        "contrast": 0.5,
        "saturation": 0.5,
        "warmth": 0.5,
        "sharpness": 0.5,
        "focus": 0.5,
        "entropy": 0.5,
    }
    blurry = perception_to_targets({"blurry": 100, "sharpness": 50}, base_features)
    crisp = perception_to_targets({"blurry": 0, "sharpness": 50}, base_features)

    assert blurry["sharpness"] < base_features["sharpness"]
    assert crisp["sharpness"] > base_features["sharpness"]
