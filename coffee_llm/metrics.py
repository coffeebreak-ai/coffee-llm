"""Замер двух решений одной задачи на одном закоммиченном дереве. В обычный маршрут не входит."""
import hashlib
import json
import subprocess
import uuid
from pathlib import Path

_FIELDS = ("input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens")


def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
    )


def require_committed(repo):
    if not repo:
        raise ValueError("не задан репозиторий замера")
    probe = _git(repo, "rev-parse", "--is-inside-work-tree")
    if probe.returncode != 0 or probe.stdout.strip() != "true":
        raise ValueError("репозиторий замера не является деревом git")
    status = _git(repo, "status", "--porcelain")
    if status.returncode != 0:
        raise ValueError("не удалось прочитать состояние репозитория")
    if status.stdout.strip():
        raise ValueError("репозиторий замера содержит незакоммиченные правки")


def is_committed(repo):
    try:
        require_committed(repo)
    except ValueError:
        return False
    return True


def restore_committed(repo):
    reset = _git(repo, "reset", "--hard", "HEAD")
    if reset.returncode != 0:
        raise ValueError("не удалось вернуть закоммиченное дерево")
    clean = _git(repo, "clean", "-fd")
    if clean.returncode != 0:
        raise ValueError("не удалось убрать незакоммиченные файлы")


def _sum(records):
    totals = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        for key in _FIELDS:
            value = record.get(key)
            if isinstance(value, bool) or not isinstance(value, int):
                continue
            totals[key] = totals.get(key, 0) + value
    return totals


def _rows(records, role):
    return [record for item_role, record in records if item_role == role]


def comparison(direct_records, normal_records):
    return {
        "expert_direct": _sum(_rows(direct_records, "expert")),
        "expert": _sum(_rows(normal_records, "expert")),
        "assistant": _sum(_rows(normal_records, "assistant")),
    }


def comparison_note(report):
    parts = []
    for name in ("expert_direct", "expert", "assistant"):
        fields = report.get(name) or {}
        chunk = [name]
        for key in _FIELDS:
            if key in fields:
                chunk.append(f"{key}={fields[key]}")
        parts.append(" ".join(chunk))
    return " | ".join(parts)


def tool_calls(items):
    return [item for item in items if isinstance(item, dict) and item.get("type") == "function_call"]


def unseen_outputs(incoming, seen):
    seen_ids = {
        item.get("call_id") for item in seen
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    }
    return [
        item for item in incoming
        if isinstance(item, dict) and item.get("type") == "function_call_output" and item.get("call_id") not in seen_ids
    ]


def measurement_path(directory, scope):
    folder = Path(directory) / "measurements"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / (hashlib.sha256(scope.encode()).hexdigest() + ".json")


def load_measurement(directory, scope):
    path = measurement_path(directory, scope)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_measurement(directory, scope, run):
    measurement_path(directory, scope).write_text(json.dumps(run, ensure_ascii=False), encoding="utf-8")


def new_measurement(items, tools):
    return {
        "phase": "direct",
        "original_items": items,
        "original_tools": tools,
        "history": json.loads(json.dumps(items)),
        "pending": [],
        "inflight": "",
        "direct_records": [],
        "normal_records": [],
        "normal_seen": None,
    }


def take_direct_results(run, incoming):
    pending = set(run["pending"])
    for item in unseen_outputs(incoming, run["history"]):
        if item.get("call_id") in pending:
            run["history"].append(item)
            pending.discard(item["call_id"])
    run["pending"] = [call_id for call_id in run["pending"] if call_id in pending]
    return [
        item for item in run["history"]
        if item.get("type") == "function_call" and item.get("call_id") in set(run["pending"])
    ]


def continue_comparison(directory, scope, items, tools, *, repository, plain_run, collected_run, expert_turn, log_note, store_report):
    """Продолжает сохранённый замер: прямой эксперт, откат дерева, обычный ход."""
    run = load_measurement(directory, scope)
    if run is None or run.get("phase") == "done":
        if not is_committed(repository):
            log_note("metrics skipped")
            return plain_run(scope, items, tools)
        run = new_measurement(items, tools)
        save_measurement(directory, scope, run)
    if run["phase"] == "direct":
        return _comparison_direct(
            directory, scope, run, items, repository,
            collected_run, expert_turn, log_note, store_report,
        )
    return _comparison_normal(directory, scope, run, items, collected_run, log_note, store_report)


def _comparison_direct(directory, scope, run, items, repository, collected_run, expert_turn, log_note, store_report):
    waiting = take_direct_results(run, items)
    if waiting:
        save_measurement(directory, scope, run)
        return waiting
    if not run["inflight"]:
        run["inflight"] = "metrics-" + uuid.uuid4().hex
        save_measurement(directory, scope, run)
    turn, records = expert_turn(run["history"], run["original_tools"], run["inflight"])
    run["direct_records"].extend(records)
    run["inflight"] = ""
    output = turn["output"]
    run["history"].extend(output)
    calls = tool_calls(output)
    if calls:
        run["pending"] = [item["call_id"] for item in calls]
        save_measurement(directory, scope, run)
        return calls
    restore_committed(repository)
    run["phase"] = "normal"
    run["pending"] = []
    run["normal_seen"] = json.loads(json.dumps(run["original_items"]))
    save_measurement(directory, scope, run)
    return _comparison_normal(
        directory, scope, run, run["original_items"],
        collected_run, log_note, store_report,
    )


def _comparison_normal(directory, scope, run, items, collected_run, log_note, store_report):
    if run["normal_seen"] is None:
        incoming = run["original_items"]
    else:
        incoming = run["normal_seen"] + unseen_outputs(items, run["normal_seen"])
    result, records = collected_run(scope + ":metrics-normal", incoming, run["original_tools"])
    run["normal_records"].extend(records)
    run["normal_seen"] = incoming
    if tool_calls(result):
        save_measurement(directory, scope, run)
        return result
    run["phase"] = "done"
    report = comparison(run["direct_records"], run["normal_records"])
    store_report(report)
    log_note(comparison_note(report))
    save_measurement(directory, scope, run)
    return result


def run_pair(repo, direct, normal):
    """Сначала прямой эксперт, затем возврат к коммиту, затем обычный ход."""
    require_committed(repo)
    try:
        direct_records = direct()
    finally:
        restore_committed(repo)
    normal_records, normal_result = normal()
    return comparison(direct_records, normal_records), normal_result
