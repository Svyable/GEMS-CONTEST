import json
from pathlib import Path

import pytest

from gems.ablation import build_spatial_ablation_commands, summarize_spatial_ablation

MATRIX = {
    "experiment_id": "ms-edge-01",
    "folds": [0, 1],
    "buffer_pixels": 16,
    "overlap": 64,
    "variants": {
        "control": {
            "config": "configs/resnet18_fold0.yaml",
            "registration_sensitivity": True,
        },
        "gradient": {
            "config": "configs/resnet18_gradient.yaml",
            "registration_sensitivity": False,
        },
        "ms_edge": {
            "config": "configs/resnet18_ms_edge.yaml",
            "registration_sensitivity": False,
        },
    },
}


def test_matrix_builds_all_variant_fold_commands(tmp_path):
    planned = build_spatial_ablation_commands(
        MATRIX,
        features="features.tif",
        labels="labels.tif",
        template="template.tif",
        fold_map="folds.tif",
        output_root=str(tmp_path),
    )
    assert len(planned) == 6
    assert [(item["variant"], item["fold"]) for item in planned] == [
        ("control", 0),
        ("control", 1),
        ("gradient", 0),
        ("gradient", 1),
        ("ms_edge", 0),
        ("ms_edge", 1),
    ]
    control = planned[0]
    assert "--registration-sensitivity-json" in control["command"]
    assert control["registration_sensitivity"] == (
        tmp_path / "ms-edge-01/control/fold-0-registration.json"
    )
    assert "--registration-sensitivity-json" not in planned[2]["command"]
    assert planned[-1]["metrics"] == tmp_path / "ms-edge-01/ms_edge/fold-1.json"


def test_summary_reports_paired_deltas(tmp_path):
    planned = build_spatial_ablation_commands(
        MATRIX,
        features="features.tif",
        labels="labels.tif",
        template="template.tif",
        fold_map="folds.tif",
        output_root=str(tmp_path),
    )
    values = {
        ("control", 0): 0.10,
        ("control", 1): 0.20,
        ("gradient", 0): 0.11,
        ("gradient", 1): 0.18,
        ("ms_edge", 0): 0.13,
        ("ms_edge", 1): 0.22,
    }
    for item in planned:
        path = Path(item["metrics"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "holdout_distance_weighted_tversky": values[
                        (item["variant"], item["fold"])
                    ]
                }
            )
        )

    summary = summarize_spatial_ablation(planned)
    assert summary["folds"] == [0, 1]
    assert summary["variants"]["control"]["mean_paired_delta_vs_baseline"] == 0
    assert summary["variants"]["gradient"]["wins_vs_baseline"] == 1
    assert summary["variants"]["gradient"]["losses_vs_baseline"] == 1
    assert summary["variants"]["ms_edge"]["wins_vs_baseline"] == 2
    assert summary["variants"]["ms_edge"]["mean_paired_delta_vs_baseline"] == pytest.approx(0.025)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda matrix: matrix.update(folds=[]),
        lambda matrix: matrix.update(variants={}),
        lambda matrix: matrix.update(experiment_id=""),
    ],
)
def test_invalid_matrix_rejected(mutator, tmp_path):
    matrix = {
        **MATRIX,
        "folds": list(MATRIX["folds"]),
        "variants": dict(MATRIX["variants"]),
    }
    mutator(matrix)
    with pytest.raises(ValueError):
        build_spatial_ablation_commands(
            matrix,
            features="features.tif",
            labels="labels.tif",
            template="template.tif",
            fold_map="folds.tif",
            output_root=str(tmp_path),
        )
