"""Schema checks for the reduced fault/trace CV training config.

configs/resnet18_cv_reduced.yaml is the scheme-agnostic baseline for
fault-discovery and trace-completion fold training on the torch+data box.
These tests pin the contract the candidate trainer (scripts/train_full_map.py)
reads directly, and the model-selection guarantees the gate relies on:
fold-pure normalization, the competition Tversky weights, and a recipe that
matches configs/resnet18_fold0.yaml except for the documented reductions.
"""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/resnet18_cv_reduced.yaml"
CONTROL = "configs/resnet18_fold0.yaml"


@pytest.fixture(scope="module")
def config():
    return yaml.safe_load((ROOT / CONFIG).read_text())


@pytest.fixture(scope="module")
def control():
    return yaml.safe_load((ROOT / CONTROL).read_text())


def test_required_trainer_keys_present(config):
    # scripts/train_full_map.py reads these keys directly from the config.
    assert int(config["patches"]["patch_size"]) == 128
    assert int(config["patches"]["train_step"]) == 64
    assert int(config["training"]["epochs"]) == 2
    assert int(config["training"]["batch_size"]) == 32
    assert float(config["training"]["learning_rate"]) > 0
    assert float(config["training"]["alpha"]) == pytest.approx(0.2)
    assert float(config["training"]["beta"]) == pytest.approx(0.8)
    assert config["model"]["encoder"] == "resnet18"
    assert config["model"]["encoder_weights"] == "imagenet"
    assert int(config["model"]["classes"]) == 1
    crop = config["augmentation"]["random_resized_crop"]
    assert len(tuple(crop["scale"])) == 2 and len(tuple(crop["ratio"])) == 2
    for key in (
        "horizontal_flip_probability",
        "vertical_flip_probability",
        "random_rotation_degrees",
    ):
        assert float(config["augmentation"][key]) >= 0


def test_loss_matches_competition_metric(config):
    # The scorer rewards recall-heavy Tversky with alpha=0.2, beta=0.8.
    assert config["training"]["loss"] == "tversky"
    assert float(config["training"]["alpha"]) == pytest.approx(0.2)
    assert float(config["training"]["beta"]) == pytest.approx(0.8)


def test_normalization_is_fold_pure(config):
    # Full-raster min/max leaks global feature statistics across folds; the
    # reduced CV baseline must fit channel ranges on the training region only.
    normalization = config["data"]["normalization"]
    assert "training_region" in normalization
    assert "full_raster" not in normalization


def _mapping_keys(node):
    if isinstance(node, dict):
        for key, value in node.items():
            yield str(key)
            yield from _mapping_keys(value)
    elif isinstance(node, list):
        for item in node:
            yield from _mapping_keys(item)


def test_scheme_comes_from_cli_not_config(config):
    # --cv-scheme, --fold, and --fold-map are CLI args pinned per run in the
    # experiment manifest; baking them into the config would break the
    # reproducible commit+config+command contract.
    keys = {k.lower().replace("-", "_") for k in _mapping_keys(config)}
    assert not ({"cv_scheme", "fold_map", "fold"} & keys)


def test_recipe_matches_control_except_reductions(config, control):
    for section in ("model", "data", "augmentation"):
        assert config[section] == control[section], section
    assert config["patches"]["patch_size"] == control["patches"]["patch_size"]
    # Documented reductions only: epochs 20 -> 2, train_step 32 -> 64.
    assert int(config["training"]["epochs"]) == 2
    assert int(control["training"]["epochs"]) == 20
    assert int(config["patches"]["train_step"]) == 64
    assert int(control["patches"]["train_step"]) == 32
