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
