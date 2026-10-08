"""Tests for automated propose-and-verify experiment gate."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from gems.experiment_gate import (
    GateDecision,
    ViewComparison,
    append_to_ledger,
    bonferroni_correction,
    compare_on_view,
    evaluate_candidate,
    format_summary,
)


def _write_score_json(path: Path, folds: dict[int, float], scheme: str = "spatial") -> None:
    """Write a synthetic score JSON like score_cv.py produces."""
    data = {
        "scheme": scheme,
        "folds": [{"fold": fold, "score": score, "valid_pixels": 1000, "truth_pixels": 100}
                  for fold, score in folds.items()],
    }
    if scheme == "spatial":
        data["aggregate_score"] = np.mean(list(folds.values()))
    else:
        data["macro_mean"] = np.mean(list(folds.values()))
        data["macro_std"] = np.std(list(folds.values()), ddof=1) if len(folds) > 1 else 0.0
    
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def test_bonferroni_correction():
    """Test Bonferroni correction calculation."""
    # With 1 trial and 3 views, α=0.05 -> α'=0.05/3
    assert bonferroni_correction(0.05, 1, 3) == pytest.approx(0.05 / 3)
    
    # With 5 trials and 3 views, α=0.05 -> α'=0.05/15
    assert bonferroni_correction(0.05, 5, 3) == pytest.approx(0.05 / 15)
    
    # With 10 trials and 2 views, α=0.01 -> α'=0.01/20
    assert bonferroni_correction(0.01, 10, 2) == pytest.approx(0.01 / 20)


def test_compare_on_view_clear_win():
    """Test view comparison when candidate clearly beats incumbent."""
    incumbent = {0: 0.5, 1: 0.52, 2: 0.48, 3: 0.51}
    candidate = {0: 0.6, 1: 0.62, 2: 0.58, 3: 0.61}  # ~0.1 better on all folds
    
    comparison = compare_on_view(incumbent, candidate, "test_view", significance_level=0.05)
    
    assert comparison.view_name == "test_view"
    assert comparison.mean_difference > 0
    assert comparison.candidate_wins  # Should win with such large differences
    assert comparison.p_value_one_sided < 0.05
    assert comparison.degrees_of_freedom == 3


def test_compare_on_view_clear_loss():
    """Test view comparison when candidate loses to incumbent."""
    incumbent = {0: 0.6, 1: 0.62, 2: 0.58, 3: 0.61}
    candidate = {0: 0.5, 1: 0.52, 2: 0.48, 3: 0.51}  # ~0.1 worse
    
    comparison = compare_on_view(incumbent, candidate, "test_view", significance_level=0.05)
    
    assert comparison.view_name == "test_view"
    assert comparison.mean_difference < 0
    assert not comparison.candidate_wins
    assert comparison.p_value_one_sided > 0.5  # One-sided test, expecting improvement


def test_compare_on_view_tie():
    """Test view comparison when scores are identical."""
    incumbent = {0: 0.5, 1: 0.5, 2: 0.5}
    candidate = {0: 0.5, 1: 0.5, 2: 0.5}
    
    comparison = compare_on_view(incumbent, candidate, "test_view", significance_level=0.05)
    
    assert comparison.mean_difference == 0.0
    assert not comparison.candidate_wins
    assert comparison.p_value_one_sided == pytest.approx(0.5)


def test_compare_on_view_partial_overlap():
    """Test that only common folds are compared."""
    incumbent = {0: 0.5, 1: 0.52, 2: 0.48}
    candidate = {1: 0.60, 2: 0.56, 3: 0.58}  # fold 3 not in incumbent
    
    comparison = compare_on_view(incumbent, candidate, "test_view", significance_level=0.05)
    
    # Should only compare folds 1 and 2
    assert len(comparison.paired_differences) == 2
    assert 1 in comparison.paired_differences
    assert 2 in comparison.paired_differences
    assert 3 not in comparison.paired_differences


def test_compare_on_view_insufficient_folds():
    """Test that error is raised with too few common folds."""
    incumbent = {0: 0.5}
    candidate = {0: 0.6}
    
    with pytest.raises(ValueError, match="need at least 2 common folds"):
        compare_on_view(incumbent, candidate, "test_view")


def test_compare_on_view_statistical_edge_cases():
    """Test statistical edge cases."""
    # Small improvement, likely not significant
    incumbent = {0: 0.50, 1: 0.51, 2: 0.49}
    candidate = {0: 0.51, 1: 0.52, 2: 0.50}  # +0.01 on all
    
    comparison = compare_on_view(incumbent, candidate, "test_view", significance_level=0.05)
    
    assert comparison.mean_difference == pytest.approx(0.01, abs=1e-10)
    # With only 3 folds and small effect, may or may not be significant
    assert comparison.p_value_one_sided >= 0  # Valid p-value


def test_evaluate_candidate_accept():
    """Test accepting a candidate that wins on 2/3 views."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        
        # Create score files
        inc_spatial = tmp / "inc_spatial.json"
        cand_spatial = tmp / "cand_spatial.json"
        inc_fault = tmp / "inc_fault.json"
        cand_fault = tmp / "cand_fault.json"
        inc_trace = tmp / "inc_trace.json"
        cand_trace = tmp / "cand_trace.json"
        
        # Incumbent baseline
        _write_score_json(inc_spatial, {0: 0.50, 1: 0.52, 2: 0.48, 3: 0.51}, "spatial")
        _write_score_json(inc_fault, {0: 0.45, 1: 0.47, 2: 0.43, 3: 0.46}, "fault")
        _write_score_json(inc_trace, {0: 0.40, 1: 0.42, 2: 0.38, 3: 0.41}, "trace")
        
        # Candidate: wins on spatial and fault, loses on trace
        _write_score_json(cand_spatial, {0: 0.60, 1: 0.62, 2: 0.58, 3: 0.61}, "spatial")  # +0.1
        _write_score_json(cand_fault, {0: 0.55, 1: 0.57, 2: 0.53, 3: 0.56}, "fault")     # +0.1
        _write_score_json(cand_trace, {0: 0.35, 1: 0.37, 2: 0.33, 3: 0.36}, "trace")     # -0.05
        
        view_scores = {
            "spatial": (inc_spatial, cand_spatial),
            "fault": (inc_fault, cand_fault),
            "trace": (inc_trace, cand_trace),
        }
        
        ledger_path = tmp / "ledger.json"
        
        decision = evaluate_candidate(
            "exp-baseline",
            "exp-candidate",
            view_scores,
            ledger_path,
            base_alpha=0.05,
            min_views_to_win=2,
        )
        
        assert decision.incumbent_id == "exp-baseline"
        assert decision.candidate_id == "exp-candidate"
        assert len(decision.view_comparisons) == 3
        assert decision.views_won == 2  # spatial and fault
        assert decision.views_required == 2
        assert decision.accepted
        assert "Candidate wins 2/3 views" in decision.reason


def test_evaluate_candidate_reject():
    """Test rejecting a candidate that wins on only 1/3 views."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        
        # Create score files
        inc_spatial = tmp / "inc_spatial.json"
        cand_spatial = tmp / "cand_spatial.json"
        inc_fault = tmp / "inc_fault.json"
        cand_fault = tmp / "cand_fault.json"
        inc_trace = tmp / "inc_trace.json"
        cand_trace = tmp / "cand_trace.json"
        
        # Incumbent baseline
        _write_score_json(inc_spatial, {0: 0.50, 1: 0.52, 2: 0.48}, "spatial")
        _write_score_json(inc_fault, {0: 0.45, 1: 0.47, 2: 0.43}, "fault")
        _write_score_json(inc_trace, {0: 0.40, 1: 0.42, 2: 0.38}, "trace")
        
        # Candidate: wins only on spatial, loses on fault and trace
        _write_score_json(cand_spatial, {0: 0.60, 1: 0.62, 2: 0.58}, "spatial")  # +0.1
        _write_score_json(cand_fault, {0: 0.40, 1: 0.42, 2: 0.38}, "fault")     # -0.05
        _write_score_json(cand_trace, {0: 0.35, 1: 0.37, 2: 0.33}, "trace")     # -0.05
        
        view_scores = {
            "spatial": (inc_spatial, cand_spatial),
            "fault": (inc_fault, cand_fault),
            "trace": (inc_trace, cand_trace),
        }
        
        ledger_path = tmp / "ledger.json"
        
        decision = evaluate_candidate(
            "exp-baseline",
            "exp-candidate",
            view_scores,
            ledger_path,
            base_alpha=0.05,
            min_views_to_win=2,
        )
        
        assert decision.views_won == 1  # only spatial
        assert not decision.accepted
        assert "wins only 1/3 views" in decision.reason


def test_append_to_ledger():
    """Test appending decisions to ledger."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger_path = Path(tmpdir) / "ledger.json"
        
        # Create a dummy decision
        comparison = ViewComparison(
            view_name="test",
            incumbent_folds={0: 0.5},
            candidate_folds={0: 0.6},
            paired_differences={0: 0.1},
            mean_difference=0.1,
            std_error=0.0,
            t_statistic=float("inf"),
            p_value_one_sided=0.0,
            degrees_of_freedom=0,
            candidate_wins=True,
        )
        
        decision = GateDecision(
            timestamp="2026-10-08T00:00:00Z",
            incumbent_id="exp-001",
            candidate_id="exp-002",
            view_comparisons=[comparison],
            views_won=1,
            views_required=1,
            trials_so_far=1,
            significance_threshold=0.05,
            accepted=True,
            reason="test",
        )
        
        # First append creates ledger
        append_to_ledger(decision, ledger_path)
        assert ledger_path.exists()
        
        with open(ledger_path) as f:
            ledger = json.load(f)
        
        assert "trials" in ledger
        assert len(ledger["trials"]) == 1
        assert ledger["trials"][0]["candidate_id"] == "exp-002"
        
        # Second append
        decision2 = GateDecision(
            timestamp="2026-10-08T01:00:00Z",
            incumbent_id="exp-002",
            candidate_id="exp-003",
            view_comparisons=[comparison],
            views_won=0,
            views_required=1,
            trials_so_far=2,
            significance_threshold=0.025,
            accepted=False,
            reason="test2",
        )
        
        append_to_ledger(decision2, ledger_path)
        
        with open(ledger_path) as f:
            ledger = json.load(f)
        
        assert len(ledger["trials"]) == 2
        assert ledger["trials"][1]["candidate_id"] == "exp-003"


def test_format_summary():
    """Test formatting decision as human-readable summary."""
    comparison = ViewComparison(
        view_name="spatial",
        incumbent_folds={0: 0.5, 1: 0.52},
        candidate_folds={0: 0.6, 1: 0.62},
        paired_differences={0: 0.1, 1: 0.1},
        mean_difference=0.1,
        std_error=0.005,
        t_statistic=20.0,
        p_value_one_sided=0.001,
        degrees_of_freedom=1,
        candidate_wins=True,
    )
    
    decision = GateDecision(
        timestamp="2026-10-08T00:00:00Z",
        incumbent_id="exp-baseline",
        candidate_id="exp-new",
        view_comparisons=[comparison],
        views_won=1,
        views_required=1,
        trials_so_far=1,
        significance_threshold=0.05,
        accepted=True,
        reason="Wins 1/1 views",
    )
    
    summary = format_summary(decision)
    
    assert "exp-new vs exp-baseline" in summary
    assert "✓ ACCEPTED" in summary
    assert "spatial" in summary
    assert "✓ WIN" in summary
    assert "+0.100000" in summary  # mean difference formatted


def test_bonferroni_tightening():
    """Test that acceptance bar tightens with more trials."""
    # First trial
    alpha_1 = bonferroni_correction(0.05, 1, 3)
    
    # Fifth trial
    alpha_5 = bonferroni_correction(0.05, 5, 3)
    
    # Tenth trial
    alpha_10 = bonferroni_correction(0.05, 10, 3)
    
    # Should tighten (get smaller)
    assert alpha_1 > alpha_5 > alpha_10
    
    # Specific values
    assert alpha_1 == pytest.approx(0.05 / 3)
    assert alpha_5 == pytest.approx(0.05 / 15)
    assert alpha_10 == pytest.approx(0.05 / 30)


def test_multiple_trials_tighten_threshold():
    """Test that subsequent trials use tighter thresholds."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        ledger_path = tmp / "ledger.json"
        
        # Create score files for first trial
        inc_spatial = tmp / "inc_spatial.json"
        cand_spatial = tmp / "cand_spatial.json"
        _write_score_json(inc_spatial, {0: 0.50, 1: 0.52}, "spatial")
        _write_score_json(cand_spatial, {0: 0.60, 1: 0.62}, "spatial")
        
        view_scores = {"spatial": (inc_spatial, cand_spatial)}
        
        # First trial
        decision1 = evaluate_candidate(
            "exp-001", "exp-002", view_scores, ledger_path, min_views_to_win=1
        )
        append_to_ledger(decision1, ledger_path)
        
        threshold_1 = decision1.significance_threshold
        
        # Second trial (reusing same scores for simplicity)
        cand2_spatial = tmp / "cand2_spatial.json"
        _write_score_json(cand2_spatial, {0: 0.61, 1: 0.63}, "spatial")
        view_scores2 = {"spatial": (inc_spatial, cand2_spatial)}
        
        decision2 = evaluate_candidate(
            "exp-002", "exp-003", view_scores2, ledger_path, min_views_to_win=1
        )
        append_to_ledger(decision2, ledger_path)
        
        threshold_2 = decision2.significance_threshold
        
        # Second trial should have tighter threshold
        assert threshold_2 < threshold_1
        assert threshold_1 == pytest.approx(0.05 / 1)  # First trial
        assert threshold_2 == pytest.approx(0.05 / 2)  # Second trial
