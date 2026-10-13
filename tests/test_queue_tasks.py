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
    assert '--seed "$SEED"' in text or "--seed $SEED" in text, \
        "train_folds.sh does not forward SEED to train_full_map.py"


def test_seed_probe_tasks_present():
    """Step-8 seed-robustness probe: spatial incumbent re-run at seed 20260923."""
    by_id = {t["id"]: t for t in TASKS}
    train = by_id.get("s8-train-seed20260923-spatial")
    assert train is not None, "s8-train-seed20260923-spatial missing"
    assert "20260923" in train["cmd"], "seed probe train task must pass seed 20260923"
    assert "configs/reference_unet.yaml" in train["cmd"], "probe must use the incumbent config"
    assert "s8-score-seed20260923-spatial" in by_id, "s8 score task missing"
    assert "s8-record-seed20260923-spatial" in by_id, "s8 record task missing"
