"""Propose-and-verify acceptance gate for GEMS experiment candidates.

Strategy priority #2 (docs/STRATEGY.md): accept a candidate (feature channel,
loss, architecture, post-processing) only if it beats the incumbent on at
least two of the three CV views (spatial-block, fault-discovery,
trace-completion) by more than the fold-to-fold standard deviation.

The verifier is `scripts/score_cv.py` output: JSON dicts with ``scheme``,
``macro_mean``, ``macro_std`` and per-fold ``folds`` entries. Both runs must
use the same metric and fold map so per-fold scores are comparable.

Search-pressure escalation: the required margin grows with the number of
already-recorded trials, because each new comparison is another chance for CV
noise to look like a real gain. With n trials already on record:

    required_margin(view) = max(incumbent_std, candidate_std)
                           * (1 + 0.5 * log2(1 + n))

Using the max of the two fold-to-fold standard deviations is a deliberate,
documented, conservative choice: a candidate that only wins by squeezing the
incumbent's noise, or by lowering its own fold variance without moving the
mean, does not pass. This escalation schedule is a judgment call, not a
theorem; its exact parameters live in the trial ledger so the bar is auditable.

Verdicts:

- ACCEPT: >= 2 comparable views won.
- REJECT: >= 2 comparable views and fewer than 2 won (enough evidence to stop).
- INCONCLUSIVE: fewer than 2 comparable views (cannot decide; do not accept).

Every call is appended to a JSONL trial ledger (both winners and losers),
which is the honest failure log required by docs/OPENAI_MATH_LEADS.md.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

VIEWS = ("spatial", "fault", "trace")

VIEW_FILES = {
    "spatial": "spatial.json",
    "fault": "fault.json",
    "trace": "trace.json",
}

ACCEPT = "ACCEPT"
REJECT = "REJECT"
INCONCLUSIVE = "INCONCLUSIVE"


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
    fold_ids: tuple[int, ...] = ()
    delta: float | None = None
    required_margin: float | None = None
    won: bool = False


def _parse_view_result(view: str, result: object) -> tuple[dict[int, float], float, float]:
    """Extract {fold_id: score}, macro_mean, macro_std from a score_cv.py JSON dict."""
    if not isinstance(result, dict):
        raise TypeError(f"{view}: expected a JSON object, got {type(result).__name__}")
    for key in ("macro_mean", "macro_std", "folds"):
        if key not in result:
            raise ValueError(f"{view}: result is missing '{key}'")
    folds: dict[int, float] = {}
    for entry in result["folds"]:
        fold = int(entry["fold"])
        score = float(entry["score"])
        if fold in folds:
            raise ValueError(f"{view}: duplicate fold {fold}")
        if not math.isfinite(score):
            raise ValueError(f"{view}: fold {fold} has a non-finite score")
        folds[fold] = score
    macro_mean = float(result["macro_mean"])
    macro_std = float(result["macro_std"])
    if not math.isfinite(macro_mean) or not math.isfinite(macro_std) or macro_std < 0:
        raise ValueError(f"{view}: macro_mean/macro_std must be finite with macro_std >= 0")
    if len(folds) < 2:
        raise ValueError(f"{view}: need at least 2 folds, found {len(folds)}")
    return folds, macro_mean, macro_std


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
        inc_folds, inc_mean, inc_std = _parse_view_result(view, incumbent_result)
        cand_folds, cand_mean, cand_std = _parse_view_result(view, candidate_result)
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
    margin = required_margin(inc_std, cand_std, trials_before)
    delta = cand_mean - inc_mean
    won = delta > margin
    return ViewComparison(
        view=view,
        comparable=True,
        incumbent_mean=inc_mean,
        incumbent_std=inc_std,
        candidate_mean=cand_mean,
        candidate_std=cand_std,
        fold_ids=tuple(sorted(inc_folds)),
        delta=delta,
        required_margin=margin,
        won=won,
    )


def verdict(comparisons: list[ViewComparison]) -> str:
    """Return ACCEPT / REJECT / INCONCLUSIVE for a set of view comparisons."""
    decided = [c for c in comparisons if c.comparable]
    if len(decided) < 2:
        return INCONCLUSIVE
    return ACCEPT if sum(c.won for c in decided) >= 2 else REJECT


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
