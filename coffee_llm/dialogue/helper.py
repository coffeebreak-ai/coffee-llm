"""Поддиалог помощника: шаги, повтор вызова и остановка по лимиту."""
import copy
import urllib.error
import uuid

from coffee_llm.constants import ASSISTANT_ROLE, EXPERT_ROLE


def helper_preamble(state):
    items = []
    for message in state["history"]:
        if message.get("role") not in ("system", "developer"):
            continue
        content = message.get("content")
        if content in (EXPERT_ROLE, ASSISTANT_ROLE):
            continue
        if isinstance(content, str) and content.startswith("Подготовленный пакет"):
            continue
        items.append(copy.deepcopy(message))
    return items


def complete_task(engine, state, ident, result):
    state["subtasks"][ident]["done"] = True
    state["subtasks"][ident]["status"] = result.get("status", "stopped")
    state["history"].append(engine._result(ident, result))
    if state["level"] == 2:
        state["escalation_pending"] = True


def stopped_task(engine, task, reason):
    return {
        "status": "stopped",
        "reason": reason,
        "partial_data": engine._bounded_partial(task, engine.evidence_char_budget, engine._fit_piece),
    }


def completed_task(engine, task, text):
    packet = engine._bounded_partial(task, engine.evidence_char_budget, engine._fit_piece)
    answer = engine._fit_piece(text, "answer")
    if answer != text:
        packet["limits"].append("ответ сокращён по бюджету доказательств")
    return {"status": "completed", "answer": answer, **packet}


def dispatch_helper_call(engine, state, task, item, emitted, target):
    if item.get("name") in ("assistant_needs_expert", "writer_needs_cloud"):
        return "escalate"
    if engine._call_signature(item) in engine._outputs_by_signature(task["history"]):
        return "repeat"
    error = engine._call_error(state, item)
    if error:
        task["history"].append(engine._result(item["call_id"], {"error": error}))
        return "continue"
    emitted.append(engine._queue(state, item, target, rename=True))
    return "wait"


def stop_unfinished_subtasks(engine, state):
    for ident, task in list(state["subtasks"].items()):
        if not task["done"] and not engine._has_pending(state, ident):
            complete_task(engine, state, ident, stopped_task(engine, task, "assistant subtask did not finish"))


def step_subtasks(engine, state, emitted):
    from coffee_llm.dialogue.engine import is_human, message_text

    for ident, task in state["subtasks"].items():
        if task["done"] or engine._has_pending(state, ident):
            continue
        if any(is_human(item) for item in state["expert_deferred"]):
            complete_task(engine, state, ident, stopped_task(engine, task, "human follow-up is waiting for expert"))
            continue
        for _ in range(engine.max_steps + 1):
            if task["steps"] >= engine.max_steps:
                complete_task(engine, state, ident, stopped_task(engine, task, "assistant subtask step limit reached"))
                break
            if not task["request_id"]:
                task["request_id"] = "coffee_llm_model_" + uuid.uuid4().hex
            engine._save(state)
            engine.log("assistant", "request", "purpose=helper subtask call_id=" + task["request_id"])
            try:
                items = engine.assistant(copy.deepcopy(task["history"]), engine._assistant_tools(state), task["request_id"])
            except urllib.error.HTTPError as error:
                if getattr(error, "gateway_retry_after", None) is not None:
                    raise
                complete_task(engine, state, ident, stopped_task(engine, task, "assistant request failed: " + type(error).__name__))
                break
            except Exception as error:
                complete_task(engine, state, ident, stopped_task(engine, task, "assistant request failed: " + type(error).__name__))
                break
            task["steps"] += 1
            task["history"].extend(items)
            task["request_id"] = ""
            engine._save(state)
            calls = [item for item in items if item.get("type") == "function_call"]
            if not calls:
                text = "\n".join(message_text(item) for item in items if item.get("role") == "assistant")
                result = completed_task(engine, task, text) if text else stopped_task(engine, task, "empty assistant answer")
                complete_task(engine, state, ident, result)
                break
            for item in calls:
                outcome = dispatch_helper_call(engine, state, task, item, emitted, ident)
                if outcome == "escalate":
                    complete_task(engine, state, ident, stopped_task(engine, task, item.get("arguments", "")))
                    break
                if outcome == "repeat":
                    complete_task(engine, state, ident, stopped_task(engine, task, "повтор без новых данных"))
                    break
            if task["done"] or engine._has_pending(state, ident):
                break
