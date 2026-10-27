#!/usr/bin/env python3
"""Audit the integrity of every committed competition result.

The committed record under results/ is the evidence every headline number
and gate verdict rests on, so this audit checks that the record is
self-consistent, in one place:

- Score files (results/runs/<run>/scores/<scheme>.json, or the same file
  at the run top level for the early "-full" runs, plus the gate-view
  copies under results/runs/gate/): five folds 0-4, each score numeric in
  [0, 1]; protocol constants (alpha 0.2, beta 0.8, metric
  distance_weighted_tversky) where present; aggregate_score in [0, 1]
  where present; macro_mean / macro_std equal the fold mean / population
  std where present.
- LOFO files: thresholds and per-fold raw/calibrated scores numeric in
  [0, 1]; top-level raw_mean / calibrated_mean equal the fold means
  (population mean, the same 1e-6 convention as lofo_threshold_audit.py);
  each fold's raw_score equals the matching raw score file's fold score.
- Run manifests (results/runs/<run>/manifest.json): every recorded
  artifact exists at its recorded path with matching bytes and SHA256.
  A recorded artifact whose content is found under a different name in
  the same run dir is reported RELOCATED (content verified, path chain
  broken); one found nowhere is MISSING.
- Gate reports (results/gates/*.json): verdict in {ACCEPT, REJECT,
  INCONCLUSIVE}; escalation_multiplier equals 1 + 0.5*log2(1 +
  trials_before); per comparable view, delta equals candidate_mean -
  incumbent_mean and required_margin equals max(incumbent_std,
  candidate_std) * escalation_multiplier (the required_margin formula
  in src/gems/verification.py); won equals delta > required_margin;
  REJECT implies no view won and ACCEPT implies the spatial view won.
- Ledger (docs/candidate-trials.jsonl): every entry names a gate report
  with the same verdict and per-view numbers; gate reports with no ledger
  entry are reported.

Usage:
    uv run python scripts/audit_results.py [--root PATH]

Exit code is 0 whenever the audit could be produced, even with findings;
a malformed file, a self-inconsistent number, a manifest hash mismatch,
or a gate/ledger disagreement is a hard error (exit 2), because an
unverifiable committed record is a reproducibility failure, not a clean
bill. Missing manifests, missing or relocated artifacts, and ledger
entries without a gate report are findings (exit 0): the numbers they
cover are still present and checked, but the provenance chain around
them is broken.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

MEAN_TOLERANCE = 1e-6
ARITH_TOLERANCE = 1e-9
N_FOLDS = 5
EXPECTED_ALPHA = 0.2
EXPECTED_BETA = 0.8
EXPECTED_METRIC = "distance_weighted_tversky"
VERDICTS = {"ACCEPT", "REJECT", "INCONCLUSIVE"}


def _fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 2


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def _load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: cannot parse JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return data


def _check_folds(data: dict, path: Path, errors: list[str]) -> list[float] | None:
    folds = data.get("folds")
    if not isinstance(folds, list) or not folds:
        errors.append(f"{path}: 'folds' must be a non-empty list")
        return None
    if len(folds) != N_FOLDS:
        errors.append(f"{path}: expected {N_FOLDS} folds, found {len(folds)}")
    scores: list[float] = []
    seen_ids: set[int] = set()
    for fold in folds:
        if not isinstance(fold, dict):
            errors.append(f"{path}: each fold must be an object")
            continue
        fold_id = fold.get("fold")
        if isinstance(fold_id, int) and not isinstance(fold_id, bool):
            seen_ids.add(fold_id)
        score = fold.get("score")
        if not _is_number(score):
            errors.append(f"{path}: fold {fold_id} lacks a numeric 'score'")
            continue
        if not 0.0 <= float(score) <= 1.0:
            errors.append(f"{path}: fold {fold_id} score {score} outside [0, 1]")
        scores.append(float(score))
    if seen_ids != set(range(N_FOLDS)):
        errors.append(f"{path}: fold ids {sorted(seen_ids)} != 0..{N_FOLDS - 1}")
    return scores


def _check_protocol(data: dict, path: Path, errors: list[str]) -> None:
    protocol = data.get("evaluation_protocol")
    if not isinstance(protocol, dict):
        return
    for key, expected in (
        ("alpha", EXPECTED_ALPHA),
        ("beta", EXPECTED_BETA),
        ("metric", EXPECTED_METRIC),
    ):
        if key in protocol and protocol[key] != expected:
            errors.append(f"{path}: protocol {key} is {protocol[key]!r}, expected {expected!r}")


def _audit_raw_score(path: Path, errors: list[str], findings: list[str]) -> list[float] | None:
    try:
        data = _load_json(path)
    except (ValueError, TypeError) as exc:
        errors.append(str(exc))
        return None
    _check_protocol(data, path, errors)
    scores = _check_folds(data, path, errors)
    aggregate = data.get("aggregate_score")
    if aggregate is not None and (not _is_number(aggregate) or not 0.0 <= float(aggregate) <= 1.0):
        errors.append(f"{path}: 'aggregate_score' must be numeric in [0, 1]")
    for key, expected in (
        ("macro_mean", statistics.fmean(scores) if scores else None),
        ("macro_std", statistics.pstdev(scores) if scores else None),
    ):
        value = data.get(key)
        if value is None or expected is None:
            continue
        if not _is_number(value) or not _close(float(value), expected, MEAN_TOLERANCE):
            errors.append(f"{path}: {key} ({value}) disagrees with fold values ({expected})")
    if scores is None:
        findings.append(f"{path}: unscorable (see errors)")
    return scores


def _audit_lofo(
    path: Path,
    raw_scores: dict[tuple[str, str], list[float]],
    run_id: str,
    scheme: str,
    errors: list[str],
    findings: list[str],
) -> None:
    try:
        data = _load_json(path)
    except (ValueError, TypeError) as exc:
        errors.append(str(exc))
        return
    folds = data.get("folds")
    if not isinstance(folds, list) or not folds:
        errors.append(f"{path}: 'folds' must be a non-empty list")
        return
    raw: list[float] = []
    cal: list[float] = []
    for fold in folds:
        if not isinstance(fold, dict):
            errors.append(f"{path}: each fold must be an object")
            continue
        for key in ("threshold", "raw_score", "calibrated_score"):
            value = fold.get(key)
            if not _is_number(value) or not 0.0 <= float(value) <= 1.0:
                errors.append(f"{path}: fold {fold.get('fold')} lacks numeric {key!r} in [0, 1]")
        if _is_number(fold.get("raw_score")):
            raw.append(float(fold["raw_score"]))
        if _is_number(fold.get("calibrated_score")):
            cal.append(float(fold["calibrated_score"]))
    for key, values in (("raw_mean", raw), ("calibrated_mean", cal)):
        top = data.get(key)
        if not _is_number(top):
            errors.append(f"{path}: missing numeric top-level {key!r}")
        elif values and not _close(float(top), statistics.fmean(values), MEAN_TOLERANCE):
            errors.append(
                f"{path}: top-level {key} ({top}) disagrees with fold mean "
                f"({statistics.fmean(values)})"
            )
    counterpart = raw_scores.get((run_id, scheme))
    if counterpart is None:
        findings.append(f"{path}: MISSING_RAW — no raw score file for this LOFO run")
        return
    if len(raw) == len(counterpart):
        for fold, raw_score, expected in zip(folds, raw, counterpart):
            if not _close(raw_score, expected, MEAN_TOLERANCE):
                errors.append(
                    f"{path}: fold {fold.get('fold')} raw_score ({raw_score}) disagrees "
                    f"with raw score file ({expected})"
                )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _audit_manifest(
    run_dir: Path,
    root: Path,
    errors: list[str],
    findings: list[str],
) -> None:
    results_dir = root / "results"
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        findings.append(f"{run_dir.name}: MISSING_MANIFEST — no manifest.json in run dir")
        return
    try:
        manifest = _load_json(manifest_path)
    except (ValueError, TypeError) as exc:
        errors.append(str(exc))
        return
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        errors.append(f"{manifest_path}: 'artifacts' must be a list")
        return
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            errors.append(f"{manifest_path}: each artifact must be an object")
            continue
        recorded = artifact.get("path")
        if not isinstance(recorded, str):
            errors.append(f"{manifest_path}: artifact lacks a string 'path'")
            continue
        target = results_dir / recorded
        if target.is_file():
            if target.stat().st_size != artifact.get("bytes"):
                errors.append(
                    f"{manifest_path}: {recorded} size {target.stat().st_size} "
                    f"!= recorded {artifact.get('bytes')}"
                )
            elif _sha256_file(target) != artifact.get("sha256"):
                errors.append(f"{manifest_path}: {recorded} SHA256 mismatch with recorded hash")
            continue
        # Not at the recorded path: same content under another name in
        # the same run dir is a relocation, not a loss.
        relocated = None
        for candidate in (run_dir / Path(recorded).name, run_dir / "scores" / Path(recorded).name):
            if candidate.is_file() and _sha256_file(candidate) == artifact.get("sha256"):
                relocated = candidate
                break
        if relocated is not None:
            findings.append(
                f"{run_dir.name}: RELOCATED — {recorded} content verified at "
                f"{relocated.relative_to(root)}"
            )
        else:
            findings.append(f"{run_dir.name}: MISSING_ARTIFACT — {recorded} not in committed tree")
    config = manifest.get("config")
    if isinstance(config, dict) and isinstance(config.get("path"), str):
        config_path = root / config["path"]
        if config_path.is_file() and _sha256_file(config_path) != config.get("sha256"):
            errors.append(f"{manifest_path}: config {config['path']} SHA256 mismatch")
        elif not config_path.is_file():
            findings.append(f"{run_dir.name}: MISSING_CONFIG — {config['path']} not in repo")


def _audit_gate(path: Path, errors: list[str]) -> dict | None:
    try:
        data = _load_json(path)
    except (ValueError, TypeError) as exc:
        errors.append(str(exc))
        return None
    verdict = data.get("verdict")
    if verdict not in VERDICTS:
        errors.append(f"{path}: verdict {verdict!r} not in {sorted(VERDICTS)}")
    trials_before = data.get("trials_before")
    multiplier = data.get("escalation_multiplier")
    if _is_number(trials_before) and _is_number(multiplier):
        expected = 1.0 + 0.5 * math.log2(1 + int(trials_before))
        if not _close(float(multiplier), expected, ARITH_TOLERANCE):
            errors.append(
                f"{path}: escalation_multiplier ({multiplier}) != "
                f"1 + 0.5*log2(1 + {trials_before}) ({expected})"
            )
    views = data.get("views")
    if not isinstance(views, list) or not views:
        errors.append(f"{path}: 'views' must be a non-empty list")
        return data
    won_views: set[str] = set()
    for view in views:
        if not isinstance(view, dict):
            errors.append(f"{path}: each view must be an object")
            continue
        name = view.get("view")
        if not view.get("comparable"):
            continue
        cand = view.get("candidate_mean")
        inc = view.get("incumbent_mean")
        cand_std = view.get("candidate_std")
        inc_std = view.get("incumbent_std")
        delta = view.get("delta")
        margin = view.get("required_margin")
        won = view.get("won")
        if not all(_is_number(v) for v in (cand, inc, delta, margin, cand_std, inc_std)):
            errors.append(f"{path}: view {name} lacks numeric comparison fields")
            continue
        if not _close(float(delta), float(cand) - float(inc), ARITH_TOLERANCE):
            errors.append(
                f"{path}: view {name} delta ({delta}) != candidate - incumbent "
                f"({float(cand) - float(inc)})"
            )
        expected_margin = max(float(inc_std), float(cand_std)) * float(multiplier or 0)
        if _is_number(multiplier) and not _close(float(margin), expected_margin, ARITH_TOLERANCE):
            errors.append(
                f"{path}: view {name} required_margin ({margin}) != "
                f"max(stds) * escalation ({expected_margin})"
            )
        if won != (float(delta) > float(margin)):
            errors.append(f"{path}: view {name} won flag ({won}) disagrees with delta/margin")
        if won:
            won_views.add(str(name))
    if verdict == "REJECT" and won_views:
        errors.append(f"{path}: verdict REJECT but views won: {sorted(won_views)}")
    if verdict == "ACCEPT" and "spatial" not in won_views:
        errors.append(f"{path}: verdict ACCEPT but the spatial view did not win")
    return data


def _audit_ledger(
    root: Path,
    gates: dict[str, dict],
    errors: list[str],
    findings: list[str],
) -> None:
    ledger_path = root / "docs" / "candidate-trials.jsonl"
    seen: set[str] = set()
    lines: list[str] = []
    if not ledger_path.is_file():
        findings.append(f"{ledger_path}: no candidate-trials ledger found")
    else:
        lines = ledger_path.read_text().splitlines()
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{ledger_path}:{lineno}: cannot parse JSON ({exc})")
            continue
        if not isinstance(entry, dict):
            errors.append(f"{ledger_path}:{lineno}: expected a JSON object")
            continue
        candidate = entry.get("candidate")
        if not isinstance(candidate, str):
            errors.append(f"{ledger_path}:{lineno}: entry lacks a string 'candidate'")
            continue
        seen.add(candidate)
        gate = gates.get(candidate)
        if gate is None:
            findings.append(
                f"{candidate}: ORPHAN_LEDGER_ENTRY — ledger entry with no gate report "
                f"in results/gates/"
            )
            continue
        if entry.get("verdict") != gate.get("verdict"):
            errors.append(
                f"{ledger_path}:{lineno}: {candidate} verdict ({entry.get('verdict')}) "
                f"!= gate report ({gate.get('verdict')})"
            )
        gate_views = {
            view.get("view"): view for view in gate.get("views", []) if isinstance(view, dict)
        }
        for view in entry.get("views", []):
            if not isinstance(view, dict):
                continue
            name = view.get("view")
            other = gate_views.get(name)
            if other is None:
                errors.append(
                    f"{ledger_path}:{lineno}: {candidate} view {name} missing from gate report"
                )
                continue
            for key in ("candidate_mean", "incumbent_mean", "delta"):
                if view.get(key) is None and other.get(key) is None:
                    continue
                if not _is_number(view.get(key)) or not _is_number(other.get(key)):
                    errors.append(
                        f"{ledger_path}:{lineno}: {candidate} view {name} {key} not numeric in both"
                    )
                    continue
                if not _close(float(view[key]), float(other[key]), ARITH_TOLERANCE):
                    errors.append(
                        f"{ledger_path}:{lineno}: {candidate} view {name} {key} "
                        f"({view[key]}) != gate report ({other[key]})"
                    )
            if view.get("won") != other.get("won"):
                errors.append(
                    f"{ledger_path}:{lineno}: {candidate} view {name} won flag "
                    f"({view.get('won')}) != gate report ({other.get('won')})"
                )
    for candidate in gates:
        if candidate not in seen:
            findings.append(f"{candidate}: ORPHAN_GATE — gate report with no ledger entry")


def _discover_score_files(runs_dir: Path) -> list[Path]:
    files = sorted(runs_dir.rglob("*.json"))
    return [
        path for path in files if path.name != "manifest.json" and not path.name.startswith("fold-")
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit integrity of committed competition results")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.root
    runs_dir = root / "results" / "runs"
    if not runs_dir.is_dir():
        return _fail(f"no results/runs/ directory under {root}")
    gates_dir = root / "results" / "gates"
    errors: list[str] = []
    findings: list[str] = []

    score_files = _discover_score_files(runs_dir)
    if not score_files:
        return _fail(f"no score files found under {runs_dir}")
    raw_scores: dict[tuple[str, str], list[float]] = {}
    lofo_files: list[tuple[Path, str, str]] = []
    for path in score_files:
        # Run id is the first path component under results/runs/, except
        # gate-view files, which live under results/runs/gate/<name>/.
        rel = path.relative_to(runs_dir)
        if rel.parts[0] == "gate":
            run_id = f"gate/{rel.parts[1]}"
        else:
            run_id = rel.parts[0]
        stem = path.stem
        if stem.startswith("lofo-"):
            lofo_files.append((path, run_id, stem.removeprefix("lofo-")))
            continue
        scores = _audit_raw_score(path, errors, findings)
        if scores is not None:
            raw_scores[(run_id, stem)] = scores
    for path, run_id, scheme in lofo_files:
        _audit_lofo(path, raw_scores, run_id, scheme, errors, findings)

    # Gate-view files must mirror a committed run score file exactly:
    # they are the gate's inputs, so an unmirrored copy is unverifiable.
    for path in score_files:
        rel = path.relative_to(runs_dir)
        if rel.parts[0] != "gate":
            continue
        try:
            data = _load_json(path)
            folds = data.get("folds")
            gate_scores = [float(f["score"]) for f in folds] if isinstance(folds, list) else None
        except (ValueError, TypeError) as exc:
            errors.append(str(exc))
            continue
        scheme = path.stem
        mirrored = any(
            key[1] == scheme
            and len(scores) == len(gate_scores or [])
            and all(_close(a, b, ARITH_TOLERANCE) for a, b in zip(scores, gate_scores or []))
            for key, scores in raw_scores.items()
            if not key[0].startswith("gate/")
        )
        if gate_scores is None or not mirrored:
            findings.append(f"{path}: UNMIRRORED_GATE_VIEW — no committed run score matches")

    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir() and p.name != "gate"):
        _audit_manifest(run_dir, root, errors, findings)

    gates: dict[str, dict] = {}
    if gates_dir.is_dir():
        for path in sorted(gates_dir.glob("*.json")):
            data = _audit_gate(path, errors)
            if data is not None and isinstance(data.get("candidate"), str):
                gates[data["candidate"]] = data
    _audit_ledger(root, gates, errors, findings)

    print(f"Score files audited: {len(score_files)} ({len(lofo_files)} LOFO)")
    print(
        f"Run manifests audited: {len([p for p in runs_dir.iterdir() if p.is_dir() and p.name != 'gate'])}"
    )
    print(f"Gate reports audited: {len(gates)}")
    if findings:
        print("\nFindings:")
        for finding in findings:
            print(f"  - {finding}")
    else:
        print("\nFindings: none")
    if errors:
        print("\nHard errors:")
        for error in errors:
            print(f"  - {error}")
        print(f"\nSummary: {len(findings)} findings, {len(errors)} hard errors — audit FAILED.")
        return 2
    print(f"\nSummary: {len(findings)} findings, 0 hard errors — numbers check out.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
