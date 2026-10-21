"""Queue integrity tests: tasks.json stays valid and references real repo files.

Catches the classic queue-edit failure modes: typos in task ids/commands,
configs that were renamed but not updated in the queue, and regressions in
the train_folds.sh seed plumbing (the seed-robustness probe depends on it).
"""

import json
import re
from pathlib import Path

REPO = Path(__file__).parent.parent
TASKS = json.loads((REPO / "scripts" / "queue" / "tasks.json").read_text())


def test_tasks_json_is_list_of_tasks():
    assert isinstance(TASKS, list) and TASKS, "tasks.json must be a non-empty list"
    for task in TASKS:
        assert isinstance(task.get("id"), str) and task["id"], f"task missing id: {task}"
        assert isinstance(task.get("cmd"), str) and task["cmd"], f"task {task['id']} missing cmd"


def test_task_ids_unique():
    ids = [t["id"] for t in TASKS]
    assert len(ids) == len(set(ids)), "duplicate task ids in tasks.json"


def test_referenced_configs_exist():
    missing = []
    for task in TASKS:
        for cfg in re.findall(r"configs/\S+\.ya?ml", task["cmd"]):
            if not (REPO / cfg).exists():
                missing.append(f"{task['id']}: {cfg}")
    assert not missing, f"tasks reference missing configs: {missing}"


def test_referenced_queue_scripts_exist():
    missing = []
    for task in TASKS:
        for script in re.findall(r"(?:/workspace/gems/queue|scripts/queue)/(\S+\.sh)", task["cmd"]):
            if not (REPO / "scripts" / "queue" / script).exists():
                missing.append(f"{task['id']}: {script}")
    assert not missing, f"tasks reference missing queue scripts: {missing}"


def test_train_folds_forwards_seed():
    """train_folds.sh must accept an optional 6th SEED arg and pass --seed through.

    Defaults to the incumbent seed 20260922 so every existing 5-arg call site
    behaves exactly as before.
    """
    text = (REPO / "scripts" / "queue" / "train_folds.sh").read_text()
    assert re.search(r"SEED=\$\{6:-20260922\}", text), "SEED default missing in train_folds.sh"
    assert '--seed "$SEED"' in text or "--seed $SEED" in text, (
        "train_folds.sh does not forward SEED to train_full_map.py"
    )


def test_seed_probe_tasks_present():
    """Step-8 seed-robustness probe: spatial incumbent re-run at seed 20260923."""
    by_id = {t["id"]: t for t in TASKS}
    train = by_id.get("s8-train-seed20260923-spatial")
    assert train is not None, "s8-train-seed20260923-spatial missing"
    assert "20260923" in train["cmd"], "seed probe train task must pass seed 20260923"
    assert "configs/reference_unet.yaml" in train["cmd"], "probe must use the incumbent config"
    assert "s8-score-seed20260923-spatial" in by_id, "s8 score task missing"
    assert "s8-record-seed20260923-spatial" in by_id, "s8 record task missing"


def test_seed_compare_task_present():
    """Step-14 seed-probe paired comparison: seed is the only tolerated
    protocol difference, so the probe never passes through gate.sh."""
    by_id = {t["id"]: t for t in TASKS}
    compare = by_id.get("s14-compare-seed20260923-b16")
    assert compare is not None, "s14-compare-seed20260923-b16 missing"
    assert "/workspace/gems/queue/compare_seed.sh" in compare["cmd"]
    assert "runs/gate/ref-b16" in compare["cmd"], "incumbent side is ref-b16"
    assert "runs/gate/ref-seed20260923-b16" in compare["cmd"], "candidate side is the probe gate dir"
    assert "gate.sh" not in compare["cmd"], "seed-mismatched runs must not go through gate.sh"
    assert compare.get("step") == 14


def test_wavelet_entropy_candidate_tasks_present():
    """Step-9 P2 wavelet-entropy candidate: 3 views train/score/record + gate."""
    by_id = {t["id"]: t for t in TASKS}
    for view in ("spatial", "fault", "trace"):
        for stage in ("train", "score", "record"):
            task_id = f"s9-{stage}-wavelet_entropy-{view}"
            assert task_id in by_id, f"{task_id} missing"
        train = by_id[f"s9-train-wavelet_entropy-{view}"]
        assert "configs/candidates/cand_wavelet_entropy.yaml" in train["cmd"]
        assert " 16" in train["cmd"], "P2 candidate must run at buffer 16"
        assert "20260923" not in train["cmd"], "P2 candidate keeps the incumbent seed"
    gate = by_id.get("s9-gate-wavelet_entropy-b16")
    assert gate is not None, "s9-gate-wavelet_entropy-b16 missing"
    assert "CAND-WAVELET_ENTROPY-B16" in gate["cmd"]
    assert "runs/gate/ref-b16" in gate["cmd"]


def test_background_soften_candidate_tasks_present():
    """Step-10 P3 background-softened loss candidate: 3 views train/score/record + gate."""
    by_id = {t["id"] for t in TASKS}
    ids = {t["id"]: t for t in TASKS}
    for view in ("spatial", "fault", "trace"):
        for stage in ("train", "score", "record"):
            task_id = f"s10-{stage}-loss_background_soften-{view}"
            assert task_id in by_id, f"{task_id} missing"
        train = ids[f"s10-train-loss_background_soften-{view}"]
        assert "configs/candidates/cand_loss_background_soften.yaml" in train["cmd"]
        assert " 16" in train["cmd"], "P3 candidate must run at buffer 16"
        assert "20260923" not in train["cmd"], "P3 candidate keeps the incumbent seed"
    gate = ids.get("s10-gate-loss_background_soften-b16")
    assert gate is not None, "s10-gate-loss_background_soften-b16 missing"
    assert "CAND-LOSS_BACKGROUND_SOFTEN-B16" in gate["cmd"]
    assert "runs/gate/ref-b16" in gate["cmd"]


def test_skeleton_recall_candidate_tasks_present():
    """Step-11 P5 skeleton-recall auxiliary loss candidate: 3 views train/score/record + gate."""
    by_id = {t["id"] for t in TASKS}
    ids = {t["id"]: t for t in TASKS}
    for view in ("spatial", "fault", "trace"):
        for stage in ("train", "score", "record"):
            task_id = f"s11-{stage}-loss_skeleton_recall-{view}"
            assert task_id in by_id, f"{task_id} missing"
        train = ids[f"s11-train-loss_skeleton_recall-{view}"]
        assert "configs/candidates/cand_loss_skeleton_recall.yaml" in train["cmd"]
        assert " 16" in train["cmd"], "P5 candidate must run at buffer 16"
        assert "20260923" not in train["cmd"], "P5 candidate keeps the incumbent seed"
    gate = ids.get("s11-gate-skeleton_recall-b16")
    assert gate is not None, "s11-gate-skeleton_recall-b16 missing"
    assert "CAND-SKELETON_RECALL-B16" in gate["cmd"]
    assert "runs/gate/ref-b16" in gate["cmd"]


def test_longer_schedule_candidate_tasks_present():
    """Step-12 longer-schedule reference: 3 views train/score/record + manual compare.

    The acceptance gate refuses epoch mismatches by design, so step 12 ends
    with scripts/compare_schedule.py instead of gate.sh.
    """
    by_id = {t["id"] for t in TASKS}
    ids = {t["id"]: t for t in TASKS}
    for view in ("spatial", "fault", "trace"):
        for stage in ("train", "score", "record"):
            task_id = f"s12-{stage}-longer_schedule-{view}"
            assert task_id in by_id, f"{task_id} missing"
        train = ids[f"s12-train-longer_schedule-{view}"]
        assert "configs/candidates/cand_longer_schedule.yaml" in train["cmd"]
        assert " 16" in train["cmd"], "schedule candidate must run at buffer 16"
        assert "20260923" not in train["cmd"], "schedule candidate keeps the incumbent seed"
    compare = ids.get("s12-compare-longer_schedule-b16")
    assert compare is not None, "s12-compare-longer_schedule-b16 missing"
    assert "/workspace/gems/queue/compare_schedule.sh" in compare["cmd"]
    assert "runs/gate/ref-b16" in compare["cmd"]
    assert "runs/gate/cand-longer_schedule-b16" in compare["cmd"]
    assert "gate.sh" not in compare["cmd"], "epoch-mismatched runs must not go through gate.sh"
    assert sum(1 for t in TASKS if t.get("step") == 12) == 10


def test_curved_continuation_candidate_tasks_present():
    """Step-13 P4 curved-continuation candidate: post-process × 3 views + gate.

    Post-processing candidates (like the straight-extension control in step 3)
    need no train/record tasks: they apply curved_view.sh to the incumbent's
    buffer-16 fold maps, score them, then run the standard gate.
    """
    by_id = {t["id"] for t in TASKS}
    ids = {t["id"]: t for t in TASKS}
    for view in ("spatial", "fault", "trace"):
        task_id = f"s13-curved-{view}"
        assert task_id in by_id, f"{task_id} missing"
        cmd = ids[task_id]["cmd"]
        assert "/workspace/gems/queue/curved_view.sh" in cmd
        assert f"runs/ref-cv-{view}-full" in cmd, "must post-process the b16 incumbent maps"
        assert f"runs/curved-cv-{view}-full" in cmd
        assert "runs/gate/cand-curved_continuation-b16" in cmd
    gate = ids.get("s13-gate-curved_continuation-b16")
    assert gate is not None, "s13-gate-curved_continuation-b16 missing"
    assert "CAND-CURVED_CONTINUATION-B16" in gate["cmd"]
    assert "runs/gate/ref-b16" in gate["cmd"]
    assert "configs/candidates/cand_curved_continuation.yaml" in gate["cmd"]
    assert sum(1 for t in TASKS if t.get("step") == 13) == 4


def test_train_score_record_triple_invariant():
    """Queue-wide invariant: every train task id of the form
    `<step>-train-<target>` must have matching `<step>-score-<target>` and
    `<step>-record-<target>` tasks.

    Generalizes the step-15 regression (s6-record-ref-trace-b32): the audit
    that caught the s6 gap caught two more — s4-ref-spatial-b32 and
    s5-ref-fault-b32 both shipped train+score with no record. record.sh is
    what writes the provenance manifest and exports the commit-safe artifacts,
    so an untripled train leaves its baseline unrecorded. New train tasks
    must ship with their score/record companions; this test fails first.
    """
    triple_pat = re.compile(r"^(s\d+[a-z]?)-(train|score|record)-(.+)$")
    groups = {}
    for t in TASKS:
        m = triple_pat.match(t["id"])
        if m:
            step, op, target = m.groups()
            groups.setdefault((step, target), set()).add(op)
    incomplete = {
        f"{step}-{op}-{target} missing": sorted({"train", "score", "record"} - ops)
        for (step, target), ops in groups.items()
        for op in ("train",)
        if {"train", "score", "record"} - ops
    }
    assert not incomplete, (
        "train tasks without a full train/score/record triple: "
        + "; ".join(f"{k} lacks {v}" for k, v in incomplete.items())
    )


def test_s4_s5_ref_b32_record_tasks_present():
    """Step-16 provenance fix: s4-ref-spatial-b32 and s5-ref-fault-b32 had
    train+score but no record, the same failure mode as the step-15 s6 gap.
    Both are buffer-32 incumbents feeding the CAND-STRUCTURE_TENSOR_COHERENCE-B32
    and CAND-RIDGE_VALLEY_RESPONSE-B32 gates.
    """
    by_id = {t["id"]: t for t in TASKS}
    for step, view in (("s4", "spatial"), ("s5", "fault")):
        for stage in ("train", "score", "record"):
            task_id = f"{step}-{stage}-ref-{view}-b32"
            assert task_id in by_id, f"{task_id} missing"
        record = by_id[f"{step}-record-ref-{view}-b32"]["cmd"]
        assert "/workspace/gems/queue/record.sh" in record
        assert f"ref-cv-{view}-b32" in record
        assert "configs/reference_unet.yaml" in record, "b32 refs use the incumbent config"


def test_s6_ref_trace_b32_record_task_present():
    """Step-6 buffer-32 trace reference needs the full train/score/record
    triple like every other queued reference run (s2, s5b): the record step
    writes the provenance manifest and exports the commit-safe artifacts, and
    ref-b32/trace.json is the incumbent input for both b32 lineament gates.

    Regression: the s6 block was committed with train+score but no record, and
    the committed ref-cv-fault-b32 results arrived with no manifest.json.
    """
    by_id = {t["id"]: t for t in TASKS}
    for stage in ("train", "score", "record"):
        task_id = f"s6-{stage}-ref-trace-b32"
        assert task_id in by_id, f"{task_id} missing"
    train = by_id["s6-train-ref-trace-b32"]["cmd"]
    assert "configs/reference_unet.yaml" in train, "b32 trace ref uses the incumbent config"
    assert " 32" in train, "b32 trace ref must run at buffer 32"
    assert "20260923" not in train, "b32 trace ref keeps the incumbent seed"
    score = by_id["s6-score-ref-trace-b32"]["cmd"]
    assert "runs/gate/ref-b32" in score, "ref trace@32 scores into the b32 gate dir"
    record = by_id["s6-record-ref-trace-b32"]["cmd"]
    assert "/workspace/gems/queue/record.sh" in record
    assert "ref-cv-trace-b32" in record
    assert "configs/reference_unet.yaml" in record
