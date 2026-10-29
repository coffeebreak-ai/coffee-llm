"""Adjuster decisions for initial routing, assistant iterations and escalation."""
import copy
import json
import urllib.request

from coffee_llm.constants import (
    ADJUSTER_ASSISTANT_CRITERIA,
    ADJUSTER_ASSISTANT_QUESTION,
    ADJUSTER_ESCALATION_CRITERIA,
    ADJUSTER_ESCALATION_QUESTION,
    ADJUSTER_INITIAL_CRITERIA,
    ADJUSTER_INITIAL_QUESTION,
    ADJUSTER_SHRINK_CRITERIA,
    ADJUSTER_SHRINK_QUESTION,
    EXPERT_CHARACTER,
    ASSISTANT_CHARACTER,
)
from coffee_llm.limits import JEV_STATE_CHAR_BUDGET, NIMBLE_TIMEOUT_SECONDS


def model_character(name, effort, role):
    return role.format(name=name, effort=effort)


def route_question(assistant, expert):
    return ADJUSTER_INITIAL_QUESTION + f" Ассистент: {assistant} Эксперт: {expert}"


def route_criteria(assistant, expert):
    return {
        "assistant": ADJUSTER_INITIAL_CRITERIA["assistant"] + " " + assistant,
        "expert": ADJUSTER_INITIAL_CRITERIA["expert"] + " " + expert,
    }



def escalation_question(expert):
    return ADJUSTER_ESCALATION_QUESTION + " " + expert


def escalation_criteria(expert):
    return {
        key: value + " " + expert
        for key, value in ADJUSTER_ESCALATION_CRITERIA.items()
    }


def assistant_question(assistant, expert):
    return ADJUSTER_ASSISTANT_QUESTION + f" Ассистент: {assistant} Эксперт: {expert}"


def assistant_criteria(assistant, expert):
    return {
        "continue_assistant": ADJUSTER_ASSISTANT_CRITERIA["continue_assistant"] + " " + assistant,
        "expert": ADJUSTER_ASSISTANT_CRITERIA["expert"] + " " + expert,
    }


def _encoded_state(state):
    return json.dumps(state, ensure_ascii=False)


def _clip_longest_text(state):
    found = [-1, None]

    def walk(value, path):
        if isinstance(value, str) and len(value) > found[0]:
            found[0], found[1] = len(value), path
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, (*path, index))
        elif isinstance(value, dict):
            for key, item in value.items():
                walk(item, (*path, key))

    walk(state, ())
    if found[1] is None or found[0] <= 1:
        return False
    parent = state
    for key in found[1][:-1]:
        parent = parent[key]
    text = parent[found[1][-1]]
    tail = max(1000, JEV_STATE_CHAR_BUDGET // 6)
    parent[found[1][-1]] = text[-tail:] if len(text) > tail else text[len(text) // 2:]
    return True


def _fit_adjuster_state(state):
    if len(_encoded_state(state)) <= JEV_STATE_CHAR_BUDGET:
        return state
    fitted = copy.deepcopy(state)
    fitted["truncated"] = True
    history = fitted.get("history", [])
    while len(history) > 1 and len(_encoded_state(fitted)) > JEV_STATE_CHAR_BUDGET:
        history.pop(0)
    for _ in range(32):
        encoded = _encoded_state(fitted)
        if len(encoded) <= JEV_STATE_CHAR_BUDGET or not _clip_longest_text(fitted):
            break
    return fitted


def decision_state(mode, items, tools):
    from coffee_llm.dialogue import is_human

    if mode == "shrink":
        content = items[0].get("content", "") if items else ""
        text = content if isinstance(content, str) else ""
        source_id = tools[0].get("source_id", "") if tools else ""
        shown = text
        if len(text) > 16_000:
            half = 8_000
            shown = text[:half] + "\n[середина не показана решателю]\n" + text[-half:]
        return _fit_adjuster_state({
            "mode": "shrink", "source_id": source_id, "length": len(text), "text": shown,
        })
    if mode not in ("initial", "assistant", "escalation"):
        raise ValueError("Unknown adjuster mode")
    latest_human = next((item for item in reversed(items) if is_human(item)), None)
    iteration_start = next((index for index in range(len(items) - 1, -1, -1)
                            if items[index].get("type") == "function_call"
                            or items[index].get("role") == "assistant"), len(items))
    while iteration_start > 0 and items[iteration_start - 1].get("type") == "function_call":
        iteration_start -= 1
    state = {
        "mode": mode,
        "evaluated_role": {"initial": "assistant", "assistant": "assistant",
                           "escalation": "expert"}[mode],
        "latest_human_message": latest_human if mode in ("initial", "assistant") else None,
        "recent_iteration": items[iteration_start:],
        "history": items,
        "tools": tools,
    }
    return _fit_adjuster_state(state)


def decide(url, model, mode, state, api_key="", assistant="", expert="", writer="", cloud=""):
    assistant = assistant or writer
    expert = expert or cloud
    questions = {
        "initial": ("route", route_question(assistant, expert), route_criteria(assistant, expert)),
        "assistant": ("assistant", assistant_question(assistant, expert), assistant_criteria(assistant, expert)),
        "escalation": ("escalation", escalation_question(expert), escalation_criteria(expert)),
        "shrink": ("shrink", ADJUSTER_SHRINK_QUESTION, ADJUSTER_SHRINK_CRITERIA),
    }
    try:
        name, instructions, criteria = questions[mode]
    except KeyError as error:
        raise ValueError("Unknown adjuster mode") from error
    body = {"model": model, "state": state, "questions": {name: {
        "type": "choice", "instructions": instructions, "criteria": criteria}}}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    request = urllib.request.Request(url.rstrip("/") + "/v1/systemone",
                                     data=json.dumps(body).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=NIMBLE_TIMEOUT_SECONDS) as response:
        answer = json.load(response)["answers"][name]
    action = answer.get("choice")
    if action not in criteria:
        raise ValueError("Unknown adjuster action")
    return {
        "action": action,
        "confidence": answer.get("confidence"),
        "reason": answer.get("reason") or answer.get("explanation") or "",
    }
