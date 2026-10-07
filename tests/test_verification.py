"""Tests for the propose-and-verify acceptance gate (gems.verification)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from statistics import pstdev

import pytest

from gems.verification import (
    ACCEPT,
    INCONCLUSIVE,
    REJECT,
    compare_view,
    count_trials,
    escalation_multiplier,
    load_view_results,
    record_trial,
    required_margin,
    verdict,
    verify_candidate,
)


def _result(mean: float, std: float, scores: list[float], scheme: str = "spatial") -> dict:
    # Construct an honest sample with the requested population standard deviation.
    spread = pstdev(scores)
    if spread:
        scores = [mean + (score - mean) * std / spread for score in scores]
    return {
        "scheme": scheme,
        "evaluation_protocol": {
            "schema_version": 1,
            "metric": "distance_weighted_tversky",
            "alpha": 0.2,
            "beta": 0.8,
            "radius_pixels": 3.0,
            "truth_sha256": "a" * 64,
            "fold_map_sha256": "b" * 64,
            "known_fault_exclusion_pixels": 0,
        },
        "macro_mean": mean,
        "macro_std": std,
        "folds": [
            {"fold": i, "score": s, "valid_pixels": 100, "truth_pixels": 10}
            for i, s in enumerate(scores)
        ],
    }


def test_escalation_multiplier_grows_with_trials():
    assert escalation_multiplier(0) == pytest.approx(1.0)
    assert escalation_multiplier(1) == pytest.approx(1.5)
    assert escalation_multiplier(3) == pytest.approx(2.0)
    assert escalation_multiplier(3) < escalation_multiplier(15)
    with pytest.raises(ValueError):
        escalation_multiplier(-1)


def test_required_margin_uses_max_std():
    m = required_margin(0.01, 0.05, 0)
    assert m == pytest.approx(0.05)
    m2 = required_margin(0.05, 0.01, 0)
    assert m2 == pytest.approx(0.05)
    with pytest.raises(ValueError):
        required_margin(float("nan"), 0.01, 0)
    with pytest.raises(ValueError):
        required_margin(0.01, -0.1, 0)


def test_compare_view_wins_when_delta_exceeds_margin():
    inc = _result(0.10, 0.01, [0.09, 0.11, 0.10])
    cand = _result(0.14, 0.01, [0.13, 0.15, 0.14])
    comp = compare_view("spatial", inc, cand, trials_before=0)
    assert comp.comparable
    assert comp.delta == pytest.approx(0.04)
    assert comp.required_margin == pytest.approx(0.01)
    assert comp.won


def test_compare_view_loses_when_delta_below_margin():
    inc = _result(0.10, 0.02, [0.08, 0.12, 0.10])
    cand = _result(0.11, 0.02, [0.09, 0.13, 0.11])
    comp = compare_view("spatial", inc, cand, trials_before=0)
    assert comp.comparable
    assert not comp.won  # delta 0.01 < margin 0.02


def test_compare_view_strict_inequality_at_boundary():
    inc = _result(0.10, 0.01, [0.09, 0.11, 0.10])
    cand = _result(0.11, 0.01, [0.10, 0.12, 0.11])
    comp = compare_view("spatial", inc, cand, trials_before=0)
    assert not comp.won  # delta == margin is not a win


def test_compare_view_search_pressure_rejects_marginal_win():
    inc = _result(0.10, 0.01, [0.09, 0.11, 0.10])
    cand = _result(0.115, 0.01, [0.105, 0.125, 0.115])
    fresh = compare_view("spatial", inc, cand, trials_before=0)
    pressured = compare_view("spatial", inc, cand, trials_before=7)
    assert fresh.won  # 0.015 > 0.01 * 1.0
    assert not pressured.won  # 0.015 < 0.01 * 2.5


def test_compare_view_fold_id_mismatch_not_comparable():
    inc = _result(0.10, 0.01, [0.09, 0.11, 0.10])
    cand = dict(_result(0.14, 0.01, [0.13, 0.15, 0.14]))
    cand["folds"] = [
        {**entry, "fold": i + 1}
        for i, entry in enumerate(cand["folds"])
    ]
    comp = compare_view("spatial", inc, cand, trials_before=0)
    assert not comp.comparable
    assert "fold ids differ" in comp.reason


def test_compare_view_rejects_bad_inputs_without_raising():
    comp = compare_view("spatial", {"nope": True}, None, trials_before=0)
    assert not comp.comparable
    comp = compare_view("fault", _result(0.1, 0.01, [0.1]), _result(0.2, 0.01, [0.2]), trials_before=0)
    assert not comp.comparable  # single fold is not enough


def test_verdict_thresholds():
    def comp(won, comparable=True):
        from gems.verification import ViewComparison

        return ViewComparison(view="x", comparable=comparable, won=won)

    assert verdict([comp(True), comp(True), comp(False)]) == ACCEPT
    assert verdict([comp(True), comp(True), comp(True)]) == ACCEPT
    assert verdict([comp(True), comp(False), comp(False)]) == REJECT
    assert verdict([comp(False), comp(False), comp(False)]) == REJECT
    assert verdict([comp(True)]) == INCONCLUSIVE  # fewer than 2 comparable views
    assert verdict([comp(True), comp(True, comparable=False)]) == INCONCLUSIVE
    assert verdict([comp(True), comp(False), comp(False, comparable=False)]) == INCONCLUSIVE
    assert verdict([comp(False), comp(False), comp(True, comparable=False)]) == REJECT


def test_verify_candidate_end_to_end():
    def big_win(mean_inc, mean_cand, scheme):
        inc = _result(mean_inc, 0.01, [mean_inc - 0.01, mean_inc + 0.01, mean_inc], scheme=scheme)
        cand = _result(mean_cand, 0.01, [mean_cand - 0.01, mean_cand + 0.01, mean_cand], scheme=scheme)
        return inc, cand

    inc_sp, cand_sp = big_win(0.10, 0.20, "spatial")
    inc_fa, cand_fa = big_win(0.05, 0.12, "fault")
    inc_tr, cand_tr = big_win(0.08, 0.081, "trace")
    report = verify_candidate(
        "TEST-CANDIDATE",
        {"spatial": inc_sp, "fault": inc_fa, "trace": inc_tr},
        {"spatial": cand_sp, "fault": cand_fa, "trace": cand_tr},
        trials_before=0,
        hypothesis="synthetic test",
        commit="abc123",
        config="configs/x.yaml",
        config_sha256="deadbeef",
    )
    assert report["verdict"] == ACCEPT
    assert report["candidate"] == "TEST-CANDIDATE"
    assert report["trials_before"] == 0
    assert report["escalation_multiplier"] == pytest.approx(1.0)
    assert report["timestamp_utc"]
    views = {v["view"]: v for v in report["views"]}
    assert views["spatial"]["won"] and views["fault"]["won"]
    assert not views["trace"]["won"]


def test_record_trial_appends_jsonl(tmp_path: Path):
    ledger = tmp_path / "trials.jsonl"
    assert count_trials(ledger) == 0
    report = verify_candidate(
        "C1",
        {"spatial": _result(0.1, 0.01, [0.09, 0.11, 0.10])},
        {"spatial": _result(0.2, 0.01, [0.19, 0.21, 0.20])},
        trials_before=0,
    )
    entry = record_trial(ledger, report)
    assert entry["trials_after"] == 1
    assert count_trials(ledger) == 1
    line = ledger.read_text(encoding="utf-8").strip().splitlines()[0]
    parsed = json.loads(line)
    assert parsed["candidate"] == "C1"
    # one blank line does not count as a trial
    ledger.write_text(ledger.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert count_trials(ledger) == 1


def test_load_view_results_reads_only_existing(tmp_path: Path):
    d = tmp_path / "scores"
    d.mkdir()
    payload = _result(0.1, 0.01, [0.09, 0.11, 0.10])
    (d / "spatial.json").write_text(json.dumps(payload), encoding="utf-8")
    results = load_view_results(d)
    assert list(results) == ["spatial"]
    assert results["spatial"]["macro_mean"] == pytest.approx(0.1)
    assert load_view_results(tmp_path / "missing") == {}


def _write_scores(tmp_path: Path, name: str, mean: float) -> Path:
    d = tmp_path / name
    d.mkdir()
    for view, scheme in (("spatial", "spatial"), ("fault", "fault"), ("trace", "trace")):
        payload = _result(mean, 0.01, [mean - 0.01, mean + 0.01, mean], scheme=scheme)
        (d / f"{view}.json").write_text(json.dumps(payload), encoding="utf-8")
    return d


def test_verify_candidate_cli_accepts_and_records(tmp_path: Path):
    inc_dir = _write_scores(tmp_path, "inc", 0.10)
    cand_dir = _write_scores(tmp_path, "cand", 0.25)
    ledger = tmp_path / "ledger.jsonl"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/verify_candidate.py",
            "--candidate",
            "CLI-TEST",
            "--hypothesis",
            "cli smoke test",
            "--incumbent-dir",
            str(inc_dir),
            "--candidate-dir",
            str(cand_dir),
            "--ledger",
            str(ledger),
            "--config",
            "configs/x.yaml",
        ],
        capture_output=True,
        check=False,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["verdict"] == ACCEPT
    assert report["trials_after"] == 1
    assert count_trials(ledger) == 1


def test_verify_candidate_cli_escalates_with_ledger(tmp_path: Path):
    inc_dir = _write_scores(tmp_path, "inc", 0.10)
    cand_dir = _write_scores(tmp_path, "cand", 0.12)  # delta 0.02 beats 0.01 margin at n=0
    ledger = tmp_path / "ledger.jsonl"
    base = [
        sys.executable,
        "scripts/verify_candidate.py",
        "--candidate",
        "MARGINAL",
        "--hypothesis",
        "escalation smoke test",
        "--incumbent-dir",
        str(inc_dir),
        "--candidate-dir",
        str(cand_dir),
        "--ledger",
        str(ledger),
    ]
    cwd = Path(__file__).resolve().parents[1]
    first = subprocess.run(base, capture_output=True, text=True, cwd=cwd, check=False)
    assert first.returncode == 0, first.stderr
    assert json.loads(first.stdout)["verdict"] == ACCEPT  # 0.02 > 0.01 * 1.0
    # six recorded trials -> multiplier 1 + 0.5*log2(7) ~= 2.4, margin ~0.024 > 0.02
    for _ in range(6):
        subprocess.run(base, capture_output=True, text=True, cwd=cwd, check=False)
    last = subprocess.run(base, capture_output=True, text=True, cwd=cwd, check=False)
    assert last.returncode == 0, last.stderr
    report = json.loads(last.stdout)
    assert report["trials_before"] == 7
    assert report["verdict"] == REJECT


def test_verify_candidate_cli_print_only_writes_nothing(tmp_path: Path):
    inc_dir = _write_scores(tmp_path, "inc", 0.10)
    cand_dir = _write_scores(tmp_path, "cand", 0.25)
    ledger = tmp_path / "ledger.jsonl"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/verify_candidate.py",
            "--candidate",
            "DRY",
            "--hypothesis",
            "dry run",
            "--incumbent-dir",
            str(inc_dir),
            "--candidate-dir",
            str(cand_dir),
            "--ledger",
            str(ledger),
            "--print-only",
        ],
        capture_output=True,
        check=False,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert proc.returncode == 0, proc.stderr
    assert not ledger.exists()
    assert json.loads(proc.stdout)["verdict"] == ACCEPT



def test_missing_protocol_cannot_be_promoted():
    inc = _result(0.10, 0.01, [0.09, 0.11, 0.10])
    cand = _result(0.30, 0.01, [0.29, 0.31, 0.30])
    cand.pop("evaluation_protocol")
    comp = compare_view("spatial", inc, cand, trials_before=0)
    assert not comp.comparable
    assert "re-score" in comp.reason


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("truth_sha256", "c" * 64),
        ("fold_map_sha256", "d" * 64),
        ("known_fault_exclusion_pixels", 2),
    ],
)
def test_different_evaluation_protocols_are_incomparable(field, replacement):
    inc = _result(0.10, 0.01, [0.09, 0.11, 0.10], scheme="fault")
    cand = _result(0.30, 0.01, [0.29, 0.31, 0.30], scheme="fault")
    cand["evaluation_protocol"][field] = replacement
    comp = compare_view("fault", inc, cand, trials_before=0)
    assert not comp.comparable
    assert field in comp.reason


def test_mislabeled_scheme_cannot_be_promoted():
    inc = _result(0.10, 0.01, [0.09, 0.11, 0.10])
    cand = _result(0.30, 0.01, [0.29, 0.31, 0.30], scheme="trace")
    comp = compare_view("spatial", inc, cand, trials_before=0)
    assert not comp.comparable
    assert "scheme mismatch" in comp.reason


def test_forged_macro_mean_is_incomparable():
    inc = _result(0.10, 0.01, [0.09, 0.11, 0.10])
    cand = _result(0.30, 0.01, [0.29, 0.31, 0.30])
    cand["macro_mean"] = 0.9
    comp = compare_view("spatial", inc, cand, trials_before=0)
    assert not comp.comparable
    assert "disagree" in comp.reason


def test_validation_coverage_mismatch_is_incomparable():
    inc = _result(0.10, 0.01, [0.09, 0.11, 0.10])
    cand = _result(0.30, 0.01, [0.29, 0.31, 0.30])
    cand["folds"][0]["truth_pixels"] = 11
    comp = compare_view("spatial", inc, cand, trials_before=0)
    assert not comp.comparable
    assert "coverage" in comp.reason


def test_invalid_fold_score_is_incomparable():
    inc = _result(0.10, 0.01, [0.09, 0.11, 0.10])
    cand = _result(0.30, 0.01, [0.29, 0.31, 0.30])
    cand["folds"][0]["score"] = 1.5
    comp = compare_view("spatial", inc, cand, trials_before=0)
    assert not comp.comparable
    assert "invalid score" in comp.reason


def test_protocol_mismatch_for_two_views_blocks_acceptance():
    inc = {v: _result(0.10, 0.01, [0.09, 0.11, 0.10], scheme=v)
           for v in ("spatial", "fault", "trace")}
    cand = {v: _result(0.30, 0.01, [0.29, 0.31, 0.30], scheme=v)
            for v in ("spatial", "fault", "trace")}
    for view in ("fault", "trace"):
        cand[view]["evaluation_protocol"]["fold_map_sha256"] = "f" * 64
    report = verify_candidate("blocked", inc, cand, trials_before=0)
    assert report["verdict"] == INCONCLUSIVE
    assert sum(v["comparable"] for v in report["views"]) == 1
