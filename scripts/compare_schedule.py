#!/usr/bin/env python3
"""Paired manual comparison for candidates the acceptance gate cannot score.

``scripts/verify_candidate.py`` refuses epoch mismatches by design
("training protocol differs: epochs"), so a longer-schedule reference can
never clear the gate against the 5-epoch incumbent. This script performs the
comparison the gate would do *except* the epoch field: it requires that
everything else -- evaluation protocol, seed, buffer, train_step, fold map,
and per-fold validation coverage -- matches exactly, then reports paired
fold-by-fold raw deltas plus LOFO-calibrated deltas when calibration
summaries are supplied.

usage:
    compare_schedule.py INCUMBENT_GATE_DIR CANDIDATE_GATE_DIR \
        [--incumbent-lofo DIR] [--candidate-lofo DIR]

Each gate dir holds {spatial,fault,trace}.json as written by
scripts/score_cv.py via scripts/queue/score_view.sh. Each lofo dir holds
lofo-{spatial,fault,trace}.json as written by
scripts/queue/lofo_calibrate_view.py. Exit 0 and print the comparison; exit
2 on any incomparable input. This script writes nothing (no ledger, no
verdict) -- the human reading the queue log decides whether the schedule
change earns incumbency.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gems.verification import (
    PROTOCOL_FIELDS,
    TRAINING_PROTOCOL_FIELDS,
    load_view_results,
)

VIEWS = ("spatial", "fault", "trace")


class _Incomparable(Exception):
    """Raised when the two runs cannot be honestly compared. main() maps
    this to exit code 2 with the message on stderr."""


def _load_gate_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _Incomparable(f"error: cannot read {path}: {exc}")
    if not isinstance(payload, dict):
        raise _Incomparable(f"error: {path} is not a JSON object")
    return payload


def _fold_scores(payload: dict, key: str = "score") -> dict[int, float]:
    folds = payload.get("folds")
    if not isinstance(folds, list) or not folds:
        raise _Incomparable(f"error: '{key}' view has no fold records")
    out: dict[int, float] = {}
    for entry in folds:
        fold = entry.get("fold")
        score = entry.get(key)
        if type(fold) is not int or type(score) not in (int, float):
            raise _Incomparable(f"error: malformed fold record in {payload.get('scheme')}")
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise _Incomparable(f"error: fold {fold} has an invalid {key}")
        out[fold] = float(score)
    return out


def _protocol(payload: dict) -> dict:
    proto = payload.get("evaluation_protocol")
    if not isinstance(proto, dict):
        raise _Incomparable(f"error: {payload.get('scheme')}: missing evaluation_protocol")
    missing = [k for k in PROTOCOL_FIELDS + TRAINING_PROTOCOL_FIELDS if k not in proto]
    if missing:
        raise _Incomparable(
            f"error: {payload.get('scheme')}: evaluation_protocol missing {missing}"
        )
    return proto


def _check_comparable(view: str, inc: dict, cand: dict) -> tuple[int, int]:
    """Require everything to match except epochs. Return (inc_epochs, cand_epochs)."""
    if inc.get("scheme") != view or cand.get("scheme") != view:
        raise _Incomparable(f"error: {view}: scheme mismatch between gate JSONs")
    p_inc, p_cand = _protocol(inc), _protocol(cand)
    for key in PROTOCOL_FIELDS:
        if p_inc[key] != p_cand[key]:
            raise _Incomparable(f"error: {view}: evaluation protocol differs in '{key}'")
    differing = [k for k in TRAINING_PROTOCOL_FIELDS if p_inc[k] != p_cand[k]]
    if differing != ["epochs"]:
        raise _Incomparable(
            f"error: {view}: training protocol differences must be exactly "
            f"['epochs'] (verify_candidate gate territory otherwise); got {differing}"
        )
    inc_scores, cand_scores = _fold_scores(inc), _fold_scores(cand)
    if set(inc_scores) != set(cand_scores):
        raise _Incomparable(f"error: {view}: fold ids differ between runs")
    return int(p_inc["epochs"]), int(p_cand["epochs"])


def _report_deltas(title: str, inc: dict[int, float], cand: dict[int, float]) -> None:
    folds = sorted(inc)
    deltas = [cand[f] - inc[f] for f in folds]
    mean_d = statistics.fmean(deltas)
    std_d = statistics.pstdev(deltas) if len(deltas) > 1 else 0.0
    mean_inc = statistics.fmean(inc[f] for f in folds)
    mean_cand = statistics.fmean(cand[f] for f in folds)
    print(f"  {title}:")
    print(f"    incumbent {mean_inc:.5f} -> candidate {mean_cand:.5f}")
    print(
        "    per-fold deltas: "
        + ", ".join(f"f{f}:{d:+.5f}" for f, d in zip(folds, deltas))
    )
    print(f"    mean delta {mean_d:+.5f} (fold-delta std {std_d:.5f})")


def _lofo_scores(lofo_dir: Path | None, view: str) -> dict[int, float] | None:
    if lofo_dir is None:
        return None
    path = lofo_dir / f"lofo-{view}.json"
    if not path.exists():
        return None
    payload = _load_gate_json(path)
    return _fold_scores(payload, key="calibrated_score")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("incumbent_gate_dir")
    parser.add_argument("candidate_gate_dir")
    parser.add_argument("--incumbent-lofo", default=None)
    parser.add_argument("--candidate-lofo", default=None)
    args = parser.parse_args()

    inc_results = load_view_results(args.incumbent_gate_dir)
    cand_results = load_view_results(args.candidate_gate_dir)
    inc_lofo = Path(args.incumbent_lofo) if args.incumbent_lofo else None
    cand_lofo = Path(args.candidate_lofo) if args.candidate_lofo else None

    for view in VIEWS:
        inc = inc_results.get(view)
        cand = cand_results.get(view)
        if inc is None or cand is None:
            print(f"  {view}: SKIPPED (missing gate JSON on at least one side)")
            continue
        inc_ep, cand_ep = _check_comparable(view, inc, cand)
        print(f"[{view}] incumbent {inc_ep} epochs vs candidate {cand_ep} epochs")
        _report_deltas("raw distance-weighted Tversky", _fold_scores(inc), _fold_scores(cand))
        lofo_inc = _lofo_scores(inc_lofo, view)
        lofo_cand = _lofo_scores(cand_lofo, view)
        if lofo_inc is not None and lofo_cand is not None:
            _report_deltas("LOFO-calibrated", lofo_inc, lofo_cand)
        else:
            print("  LOFO-calibrated: SKIPPED (no lofo summaries on one side)")
    return 0


def _entry() -> int:
    try:
        return main()
    except _Incomparable as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(_entry())
