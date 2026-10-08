"""Automated propose-and-verify experiment gate with multiple-comparisons correction.

Implements roadmap priority #2: accept a candidate run only if it beats the incumbent
on ≥2 of 3 CV views (spatial, fault-discovery, trace-completion) by more than
fold-to-fold noise, with a multiple-comparisons-aware acceptance bar that tightens
as more candidates are tried.

This is our defense against overfitting local CV through search pressure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class ViewComparison:
    """Comparison of candidate vs incumbent on one CV view."""

    view_name: str
    incumbent_folds: dict[int, float]
    candidate_folds: dict[int, float]
    paired_differences: dict[int, float]  # candidate - incumbent per fold
    mean_difference: float
    std_error: float
    t_statistic: float
    p_value_one_sided: float
    degrees_of_freedom: int
    candidate_wins: bool

    def to_dict(self) -> dict:
        return {
            "view_name": self.view_name,
            "incumbent_folds": self.incumbent_folds,
            "candidate_folds": self.candidate_folds,
            "paired_differences": self.paired_differences,
            "mean_difference": self.mean_difference,
            "std_error": self.std_error,
            "t_statistic": self.t_statistic,
            "p_value_one_sided": self.p_value_one_sided,
            "degrees_of_freedom": self.degrees_of_freedom,
            "candidate_wins": self.candidate_wins,
        }


@dataclass(frozen=True)
class GateDecision:
    """Result of propose-and-verify gate."""

    timestamp: str
    incumbent_id: str
    candidate_id: str
    view_comparisons: list[ViewComparison]
    views_won: int
    views_required: int
    trials_so_far: int
    significance_threshold: float
    accepted: bool
    reason: str

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "incumbent_id": self.incumbent_id,
            "candidate_id": self.candidate_id,
            "view_comparisons": [vc.to_dict() for vc in self.view_comparisons],
            "views_won": self.views_won,
            "views_required": self.views_required,
            "trials_so_far": self.trials_so_far,
            "significance_threshold": self.significance_threshold,
            "accepted": self.accepted,
            "reason": self.reason,
        }


def _load_cv_scores(path: Path) -> dict[int, float]:
    """Load fold-level scores from score_cv.py JSON output."""
    with open(path) as f:
        data = json.load(f)
    
    # Handle both spatial (has aggregate_score) and fault-discovery (has macro_mean)
    if "folds" not in data:
        raise ValueError(f"{path}: missing 'folds' key")
    
    folds = {}
    for fold_data in data["folds"]:
        fold_id = fold_data["fold"]
        score = fold_data["score"]
        folds[fold_id] = score
    
    return folds


def compare_on_view(
    incumbent_scores: dict[int, float],
    candidate_scores: dict[int, float],
    view_name: str,
    *,
    significance_level: float = 0.05,
) -> ViewComparison:
    """Compare candidate vs incumbent on one CV view using paired t-test.
    
    Args:
        incumbent_scores: Fold ID -> score mapping for incumbent
        candidate_scores: Fold ID -> score mapping for candidate
        view_name: Name of this CV view (e.g., "spatial", "fault-discovery")
        significance_level: One-sided significance level for t-test
    
    Returns:
        ViewComparison with statistical test results
    """
    # Match folds
    common_folds = sorted(set(incumbent_scores.keys()) & set(candidate_scores.keys()))
    if len(common_folds) < 2:
        raise ValueError(
            f"{view_name}: need at least 2 common folds, got {len(common_folds)}"
        )
    
    # Compute paired differences (candidate - incumbent)
    differences = {}
    for fold in common_folds:
        differences[fold] = candidate_scores[fold] - incumbent_scores[fold]
    
    # Paired t-test statistics
    diff_values = np.array([differences[f] for f in common_folds])
    n = len(diff_values)
    mean_diff = float(np.mean(diff_values))
    std_diff = float(np.std(diff_values, ddof=1))
    se_diff = std_diff / np.sqrt(n)
    df = n - 1
    
    # One-sided t-test: is candidate > incumbent?
    if se_diff > 1e-10:  # Non-zero standard error
        t_stat = mean_diff / se_diff
        p_value = 1 - stats.t.cdf(t_stat, df)  # one-sided upper tail
    else:
        # All differences identical (zero variance)
        t_stat = 0.0
        if mean_diff > 0:
            # All positive differences -> extremely significant
            p_value = 0.0
        elif mean_diff < 0:
            # All negative differences -> no evidence for improvement
            p_value = 1.0
        else:
            # Perfect tie (all zeros)
            p_value = 0.5
    
    candidate_wins = (p_value < significance_level) and (mean_diff > 0)
    
    return ViewComparison(
        view_name=view_name,
        incumbent_folds={f: incumbent_scores[f] for f in common_folds},
        candidate_folds={f: candidate_scores[f] for f in common_folds},
        paired_differences=differences,
        mean_difference=mean_diff,
        std_error=se_diff,
        t_statistic=t_stat,
        p_value_one_sided=p_value,
        degrees_of_freedom=df,
        candidate_wins=candidate_wins,
    )


def bonferroni_correction(
    base_alpha: float,
    trials_so_far: int,
    views_to_test: int = 3,
) -> float:
    """Compute Bonferroni-corrected significance threshold.
    
    With multiple comparisons across trials and views, we need to control
    the family-wise error rate. Use conservative Bonferroni correction.
    
    Args:
        base_alpha: Base significance level (e.g., 0.05)
        trials_so_far: Number of candidates tried so far (including this one)
        views_to_test: Number of views being compared (default 3)
    
    Returns:
        Corrected significance threshold
    """
    # Total comparisons = trials × views
    total_comparisons = trials_so_far * views_to_test
    return base_alpha / total_comparisons


def evaluate_candidate(
    incumbent_id: str,
    candidate_id: str,
    view_scores: dict[str, tuple[Path, Path]],
    ledger_path: Path,
    *,
    base_alpha: float = 0.05,
    min_views_to_win: int = 2,
) -> GateDecision:
    """Evaluate whether to accept a candidate run via the propose-and-verify gate.
    
    Args:
        incumbent_id: Run ID of incumbent (e.g., "exp-001")
        candidate_id: Run ID of candidate (e.g., "exp-002")
        view_scores: Dict mapping view name -> (incumbent JSON path, candidate JSON path)
                     e.g., {"spatial": (inc_spatial.json, cand_spatial.json)}
        ledger_path: Path to append-only trial ledger JSON
        base_alpha: Base significance level before correction
        min_views_to_win: Minimum views candidate must win (default: 2 of 3)
    
    Returns:
        GateDecision with accept/reject verdict and statistical details
    """
    # Load existing trial ledger to count attempts
    trials_so_far = 1  # This candidate
    if ledger_path.exists():
        with open(ledger_path) as f:
            ledger = json.load(f)
            trials_so_far = len(ledger["trials"]) + 1
    
    # Compute corrected significance threshold
    corrected_alpha = bonferroni_correction(base_alpha, trials_so_far, len(view_scores))
    
    # Compare on each view
    comparisons = []
    for view_name, (inc_path, cand_path) in view_scores.items():
        inc_scores = _load_cv_scores(inc_path)
        cand_scores = _load_cv_scores(cand_path)
        
        comparison = compare_on_view(
            inc_scores,
            cand_scores,
            view_name,
            significance_level=corrected_alpha,
        )
        comparisons.append(comparison)
    
    # Count wins
    views_won = sum(1 for c in comparisons if c.candidate_wins)
    
    # Decision: accept if wins >= min_views_to_win
    accepted = views_won >= min_views_to_win
    
    if accepted:
        reason = (
            f"Candidate wins {views_won}/{len(comparisons)} views "
            f"(required: {min_views_to_win}), each at α={corrected_alpha:.6f} "
            f"after Bonferroni correction for {trials_so_far} trials"
        )
    else:
        reason = (
            f"Candidate wins only {views_won}/{len(comparisons)} views "
            f"(required: {min_views_to_win}), rejected"
        )
    
    decision = GateDecision(
        timestamp=datetime.now(timezone.utc).isoformat(),
        incumbent_id=incumbent_id,
        candidate_id=candidate_id,
        view_comparisons=comparisons,
        views_won=views_won,
        views_required=min_views_to_win,
        trials_so_far=trials_so_far,
        significance_threshold=corrected_alpha,
        accepted=accepted,
        reason=reason,
    )
    
    return decision


def append_to_ledger(decision: GateDecision, ledger_path: Path) -> None:
    """Append decision to the trial ledger.
    
    The ledger is an append-only JSON file tracking all gate decisions.
    """
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    
    if ledger_path.exists():
        with open(ledger_path) as f:
            ledger = json.load(f)
    else:
        ledger = {
            "description": "Automated propose-and-verify experiment gate ledger",
            "base_alpha": 0.05,
            "min_views_to_win": 2,
            "trials": [],
        }
    
    ledger["trials"].append(decision.to_dict())
    
    with open(ledger_path, "w") as f:
        json.dump(ledger, f, indent=2)


def format_summary(decision: GateDecision) -> str:
    """Format a human-readable summary for docs/EXPERIMENT_LOG.md."""
    lines = []
    lines.append(f"## {decision.candidate_id} vs {decision.incumbent_id}")
    lines.append(f"**Timestamp:** {decision.timestamp}")
    lines.append(f"**Decision:** {'✓ ACCEPTED' if decision.accepted else '✗ REJECTED'}")
    lines.append(f"**Reason:** {decision.reason}")
    lines.append("")
    lines.append("### View Comparisons")
    lines.append("")
    
    for vc in decision.view_comparisons:
        status = "✓ WIN" if vc.candidate_wins else "✗ LOSS"
        lines.append(f"**{vc.view_name}:** {status}")
        lines.append(f"- Mean difference: {vc.mean_difference:+.6f} (SE: {vc.std_error:.6f})")
        lines.append(f"- t({vc.degrees_of_freedom}) = {vc.t_statistic:.3f}, p = {vc.p_value_one_sided:.6f}")
        lines.append(f"- Threshold: α = {decision.significance_threshold:.6f}")
        lines.append("")
    
    return "\n".join(lines)
