"""Select candidates by stitched, pooled spatial OOF distance-weighted Tversky.

A pooled spatial gain is mandatory. Geographic folds and overlapping-background
fault/trace views are separate non-regression safeguards, never pooled together.
All three views must be comparable before acceptance. Reports come from the
schema-2 score_cv.py scorer; legacy evidence must be re-scored.

The improvement hurdle remains max(fold stds) * (1 + 0.5 * log2(1 + trials)).
This is a conservative search-pressure heuristic, not a statistical confidence
bound for the pooled ratio. Non-regression tolerance is the un-escalated maximum
fold std: repeated trials must not make material regressions easier to tolerate.
Every verdict remains recorded in the JSONL failure ledger.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, pstdev

from gems.evaluation import AGGREGATION_METHODS
from gems.metric import TverskyComponents

VIEWS = ("spatial", "fault", "trace")

VIEW_FILES = {
    "spatial": "spatial.json",
    "fault": "fault.json",
    "trace": "trace.json",
}

ACCEPT = "ACCEPT"
REJECT = "REJECT"
INCONCLUSIVE = "INCONCLUSIVE"

PROTOCOL_FIELDS = (
    "schema_version",
    "aggregation",
    "metric",
    "alpha",
    "beta",
    "radius_pixels",
    "truth_sha256",
    "fold_map_sha256",
    "known_fault_exclusion_pixels",
)

TRAINING_PROTOCOL_FIELDS = (
    "buffer_pixels",
    "seed",
    "epochs",
    "train_step",
)

SELECTION_PROTOCOL = "pooled_spatial_with_nonregression_v2"


def escalation_multiplier(trials_before: int) -> float:
    """Return the search-pressure multiplier after ``trials_before`` recorded trials."""
    if trials_before < 0:
        raise ValueError("trials_before must be non-negative")
    return 1.0 + 0.5 * math.log2(1 + trials_before)


def required_margin(incumbent_std: float, candidate_std: float, trials_before: int) -> float:
    """Return the per-view margin a candidate must exceed to win a view.

    Margin = max(fold-to-fold std of both runs) * escalation(trials_before).
    """
    for name, value in (("incumbent_std", incumbent_std), ("candidate_std", candidate_std)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite non-negative number")
    return max(incumbent_std, candidate_std) * escalation_multiplier(trials_before)


@dataclass(frozen=True)
class ViewComparison:
    """Outcome of one CV view's incumbent-vs-candidate comparison."""

    view: str
    comparable: bool
    reason: str = ""
    incumbent_mean: float | None = None
    incumbent_std: float | None = None
    candidate_mean: float | None = None
    candidate_std: float | None = None
    score_basis: str = ""
    incumbent_score: float | None = None
    candidate_score: float | None = None
    fold_ids: tuple[int, ...] = ()
    delta: float | None = None
    required_margin: float | None = None
    regression_tolerance: float | None = None
    fold_deltas: tuple[tuple[int, float], ...] = ()
    safeguard_passed: bool = False
    won: bool = False


def _parse_view_result(
    view: str, result: object
) -> tuple[dict[int, tuple[float, int, int]], float, float, dict, float]:
    """Validate one scorer report, including its immutable evaluation protocol."""
    if not isinstance(result, dict):
        raise TypeError(f"{view}: expected a JSON object, got {type(result).__name__}")
    if result.get("scheme") != view:
        raise ValueError(f"{view}: scheme mismatch (got {result.get('scheme')!r})")

    raw_protocol = result.get("evaluation_protocol")
    if not isinstance(raw_protocol, dict):
        raise TypeError(f"{view}: missing evaluation_protocol; re-score with score_cv.py")
    missing = [key for key in PROTOCOL_FIELDS if key not in raw_protocol]
    if missing:
        raise ValueError(f"{view}: evaluation_protocol missing {', '.join(missing)}")
    protocol = {key: raw_protocol[key] for key in PROTOCOL_FIELDS}
    expected = {
        "schema_version": 2,
        "aggregation": AGGREGATION_METHODS[view],
        "metric": "distance_weighted_tversky",
        "alpha": 0.2,
        "beta": 0.8,
        "radius_pixels": 3.0,
    }
    for key, value in expected.items():
        if type(protocol[key]) is not type(value) or protocol[key] != value:
            raise ValueError(f"{view}: unsupported evaluation_protocol {key}; re-score")
    for key in ("truth_sha256", "fold_map_sha256"):
        digest = protocol[key]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"{view}: invalid evaluation_protocol {key}")
    exclusion = protocol["known_fault_exclusion_pixels"]
    if type(exclusion) is not int or exclusion < 0 or (view == "spatial" and exclusion):
        raise ValueError(f"{view}: invalid known_fault_exclusion_pixels")

    for key in ("macro_mean", "macro_std", "folds"):
        if key not in result:
            raise ValueError(f"{view}: result is missing '{key}'")
    if not isinstance(result["folds"], list):
        raise TypeError(f"{view}: folds must be a list")
    folds: dict[int, tuple[float, int, int]] = {}
    for entry in result["folds"]:
        if not isinstance(entry, dict):
            raise TypeError(f"{view}: fold record must be an object")
        fold = entry.get("fold")
        if type(fold) is not int or fold < 0:
            raise ValueError(f"{view}: invalid fold id {fold!r}")
        score = entry.get("score")
        valid_pixels = entry.get("valid_pixels")
        truth_pixels = entry.get("truth_pixels")
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"{view}: fold {fold} has an invalid score")
        if (type(valid_pixels) is not int or type(truth_pixels) is not int
                or valid_pixels <= 0 or not 0 <= truth_pixels <= valid_pixels):
            raise ValueError(f"{view}: fold {fold} has invalid pixel counts")
        if fold in folds:
            raise ValueError(f"{view}: duplicate fold {fold}")
        folds[fold] = (float(score), valid_pixels, truth_pixels)
    if len(folds) < 2:
        raise ValueError(f"{view}: need at least 2 folds, found {len(folds)}")

    raw_mean, raw_std = result["macro_mean"], result["macro_std"]
    if type(raw_mean) not in (int, float) or type(raw_std) not in (int, float):
        raise ValueError(f"{view}: macro statistics must be numeric")
    macro_mean, macro_std = float(raw_mean), float(raw_std)
    if (not math.isfinite(macro_mean) or not math.isfinite(macro_std)
            or not 0 <= macro_mean <= 1 or macro_std < 0):
        raise ValueError(f"{view}: invalid macro statistics")
    scores = [folds[fold][0] for fold in sorted(folds)]
    if (not math.isclose(macro_mean, mean(scores), abs_tol=1e-9, rel_tol=1e-9)
            or not math.isclose(macro_std, pstdev(scores), abs_tol=1e-9, rel_tol=1e-9)):
        raise ValueError(f"{view}: macro statistics disagree with per-fold scores")
    primary_score = macro_mean
    if view == "spatial":
        primary_score = _validate_spatial_aggregate(result, folds)
    # Preserve budget/split metadata rather than silently discarding it. Config
    # hashes are provenance, not equality keys: candidate configs may differ.
    for key in TRAINING_PROTOCOL_FIELDS:
        if key in raw_protocol:
            protocol[key] = raw_protocol[key]
    return folds, macro_mean, macro_std, protocol, primary_score


def _validate_spatial_aggregate(result: dict, folds: dict) -> float:
    score = result.get("aggregate_score")
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("spatial: missing or invalid aggregate_score; re-score")
    components = result.get("aggregate_components")
    if not isinstance(components, dict):
        raise TypeError("spatial: missing aggregate_components; re-score")
    for key, index in (("valid_pixels", 1), ("truth_pixels", 2)):
        count = components.get(key)
        if type(count) is not int or count != sum(f[index] for f in folds.values()):
            raise ValueError(f"spatial: aggregate {key} disagrees with fold coverage")
    for key in ("tp_w", "fp_w", "fn_w"):
        value = components.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"spatial: invalid aggregate component {key}")
    if not math.isclose(
        components["tp_w"] + components["fn_w"], components["truth_pixels"],
        abs_tol=1e-9, rel_tol=1e-9,
    ):
        raise ValueError("spatial: aggregate TP + FN disagrees with truth coverage")
    if components["fp_w"] > components["valid_pixels"]:
        raise ValueError("spatial: aggregate FP exceeds valid coverage")
    measured = TverskyComponents(*(components[k] for k in ("tp_w", "fp_w", "fn_w")))
    if not math.isclose(score, measured.score(), abs_tol=1e-9, rel_tol=1e-9):
        raise ValueError("spatial: aggregate_score disagrees with weighted components")
    return float(score)


def compare_view(
    view: str,
    incumbent_result: object,
    candidate_result: object,
    *,
    trials_before: int,
) -> ViewComparison:
    """Compare one CV view. Returns a non-comparable comparison with a reason
    instead of raising when the two runs cannot be fairly compared."""
    try:
        inc_folds, inc_mean, inc_std, inc_protocol, inc_score = _parse_view_result(
            view, incumbent_result
        )
        cand_folds, cand_mean, cand_std, cand_protocol, cand_score = _parse_view_result(
            view, candidate_result
        )
    except (ValueError, KeyError, TypeError) as exc:
        return ViewComparison(view=view, comparable=False, reason=str(exc))
    if set(inc_folds) != set(cand_folds):
        return ViewComparison(
            view=view,
            comparable=False,
            reason=(
                f"fold ids differ: incumbent {sorted(inc_folds)} "
                f"vs candidate {sorted(cand_folds)}"
            ),
        )
    differing = [key for key in PROTOCOL_FIELDS if inc_protocol[key] != cand_protocol[key]]
    if differing:
        return ViewComparison(
            view=view,
            comparable=False,
            reason="evaluation protocol differs: " + ", ".join(differing),
        )
    
    training_differing = []
    for key in TRAINING_PROTOCOL_FIELDS:
        inc_val = inc_protocol.get(key)
        cand_val = cand_protocol.get(key)
        if inc_val != cand_val:
            training_differing.append(key)
    if training_differing:
        return ViewComparison(
            view=view,
            comparable=False,
            reason="training protocol differs: " + ", ".join(training_differing),
        )
    for fold in sorted(inc_folds):
        if inc_folds[fold][1:] != cand_folds[fold][1:]:
            return ViewComparison(
                view=view,
                comparable=False,
                reason=f"fold {fold} validation pixel coverage differs",
            )
    margin = required_margin(inc_std, cand_std, trials_before)
    delta = cand_score - inc_score
    fold_deltas = tuple(
        (fold, cand_folds[fold][0] - inc_folds[fold][0]) for fold in sorted(inc_folds)
    )
    tolerance = max(inc_std, cand_std)
    safeguard_passed = delta >= -tolerance or math.isclose(
        delta, -tolerance, abs_tol=1e-12, rel_tol=1e-12
    )
    if view == "spatial":
        safeguard_passed = safeguard_passed and all(
            d >= -tolerance or math.isclose(d, -tolerance, abs_tol=1e-12, rel_tol=1e-12)
            for _, d in fold_deltas
        )
    # Floating-point rounding at an exact boundary is not a real win.
    won = delta > margin and not math.isclose(delta, margin, abs_tol=1e-12, rel_tol=1e-12)
    return ViewComparison(
        view=view,
        comparable=True,
        incumbent_mean=inc_mean,
        incumbent_std=inc_std,
        candidate_mean=cand_mean,
        candidate_std=cand_std,
        score_basis="aggregate_score" if view == "spatial" else "macro_mean",
        incumbent_score=inc_score,
        candidate_score=cand_score,
        fold_ids=tuple(sorted(inc_folds)),
        delta=delta,
        required_margin=margin,
        regression_tolerance=tolerance,
        fold_deltas=fold_deltas,
        safeguard_passed=safeguard_passed,
        won=won,
    )


def verdict(comparisons: list[ViewComparison]) -> str:
    """Require a pooled spatial win plus all three comparable safeguards."""
    by_view = {c.view: c for c in comparisons}
    spatial = by_view.get("spatial")
    if spatial is not None and spatial.comparable and not spatial.won:
        return REJECT
    if any(c.comparable and not c.safeguard_passed for c in comparisons):
        return REJECT
    if all(v in by_view and by_view[v].comparable for v in VIEWS):
        return ACCEPT
    return INCONCLUSIVE


def load_view_results(directory: str | Path) -> dict[str, dict]:
    """Load score_cv.py JSON results from a directory (spatial.json/fault.json/trace.json).

    Returns {view: result_dict} for the view files that exist.
    """
    directory = Path(directory)
    results: dict[str, dict] = {}
    for view, filename in VIEW_FILES.items():
        path = directory / filename
        if not path.exists():
            continue
        results[view] = json.loads(path.read_text(encoding="utf-8"))
    return results


def count_trials(ledger_path: str | Path) -> int:
    """Count non-empty lines in the JSONL trial ledger (0 when absent)."""
    path = Path(ledger_path)
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def verify_candidate(
    candidate_name: str,
    incumbent_results: dict[str, dict],
    candidate_results: dict[str, dict],
    *,
    trials_before: int,
    hypothesis: str = "",
    commit: str = "",
    config: str = "",
    config_sha256: str = "",
) -> dict:
    """Run the full propose-and-verify gate and return the machine-readable report."""
    comparisons = [
        compare_view(
            view,
            incumbent_results.get(view),
            candidate_results.get(view),
            trials_before=trials_before,
        )
        for view in VIEWS
    ]
    decision = verdict(comparisons)
    return {
        "selection_protocol": SELECTION_PROTOCOL,
        "candidate": candidate_name,
        "hypothesis": hypothesis,
        "commit": commit,
        "config": config,
        "config_sha256": config_sha256,
        "trials_before": trials_before,
        "escalation_multiplier": escalation_multiplier(trials_before),
        "verdict": decision,
        "views": [asdict(c) for c in comparisons],
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }


def record_trial(ledger_path: str | Path, report: dict) -> dict:
    """Append a verify_candidate report as one JSONL line; return it with trials_after."""
    path = Path(ledger_path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    entry = dict(report)
    entry["trials_after"] = entry["trials_before"] + 1
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry
