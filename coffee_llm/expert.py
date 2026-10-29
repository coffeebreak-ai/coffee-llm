"""Ход эксперта и переход со второго уровня на третий."""
import copy
import json
import uuid

from coffee_llm.assistant import validate_arguments
from coffee_llm.constants import ASSISTANT_TOOL, HELPER_ROLE
from coffee_llm.helper import helper_preamble, step_subtasks, stop_unfinished_subtasks
from coffee_llm.packet import run_prep


def clear_expert_request(state):
    state["expert_response"], state["expert_request"], state["expert_request_id"] = None, None, ""


def flush_expert_tail(engine, state):
    for item in state["expert_deferred"]:
        state["history"].append(item)
        engine._archive_client_item(state, item)
    state["expert_deferred"] = []


def evaluate_escalation(engine, state):
    if not state["escalation_pending"]:
        return
    if state["level"] == 2:
        decision = engine._decision("escalation", state["history"], state["tools"], 2)
        if decision["action"] == "expert_tools":
            state["level"] = 3
    state["escalation_pending"] = False
    engine._save(state)


def expert_tools(engine, state):
    from coffee_llm.dialogue import RESERVED_TOOLS

    frozen = state.get("level_tools")
    if isinstance(frozen, dict) and frozen.get("level") == state["level"]:
        return copy.deepcopy(frozen["tools"])
    virtual = copy.deepcopy(ASSISTANT_TOOL)
    if state["legacy_tools"]:
        virtual["name"] = "gateway_writer"
    tools = [virtual] if state["level"] == 2 else [
        *copy.deepcopy([tool for tool in state["tools"] if tool.get("name") not in RESERVED_TOOLS]), virtual]
    state["level_tools"] = {"level": state["level"], "tools": copy.deepcopy(tools)}
    return copy.deepcopy(tools)


def stable_expert_history(engine, state):
    history = state["history"]
    sent = state.get("expert_sent") or []
    if sent and history[:len(sent)] != sent:
        history = [*copy.deepcopy(sent), *history[len(sent):]]
        state["history"] = history
        engine.log("expert", "history", "prefix restored")
    state["expert_sent"] = copy.deepcopy(history)
    return copy.deepcopy(history)


def outer_expert(engine, state, emitted):
    from coffee_llm.dialogue import encoded, message_text

    for _ in range(engine.max_steps):
        if state["expert_response"] is None:
            if state["expert_request"] is None:
                evaluate_escalation(engine, state)
                state["expert_request"] = {
                    "history": stable_expert_history(engine, state),
                    "tools": expert_tools(engine, state),
                    "model": engine.expert_model,
                    "effort": engine.expert_effort,
                }
            if not state["expert_request_id"]:
                state["expert_request_id"] = "coffee_llm_model_" + uuid.uuid4().hex
            engine._save(state)
            engine.log("expert", "request", "purpose=expert call_id=" + state["expert_request_id"])
            request = state["expert_request"]
            turn = engine.expert(copy.deepcopy(request["history"]),
                                 copy.deepcopy(request["tools"]), state["expert_request_id"],
                                 model=request.get("model", engine.expert_model),
                                 effort=request.get("effort", engine.expert_effort))
            state["expert_response"] = ({"status": turn.get("status", "completed"), "output": turn.get("output", [])}
                                        if isinstance(turn, dict) else {"status": "completed", "output": turn})
            engine._save(state)
        response = state["expert_response"]
        items = response["output"]
        if not isinstance(items, list):
            return engine._pause(state)
        state["history"].extend(items)
        calls = [item for item in items if item.get("type") == "function_call"]
        if not calls:
            clear_expert_request(state)
            if state["expert_deferred"]:
                flush_expert_tail(engine, state)
                engine._save(state)
                continue
            if response["status"] == "completed" and any(item.get("role") == "assistant" and message_text(item) for item in items):
                result = engine._finish(state, items)
                engine._save(state)
                return result
            engine._save(state)
            return engine._pause(state)
        for item in calls:
            virtual_names = {"gateway_assistant"}
            if state["legacy_tools"]:
                virtual_names.add("gateway_writer")
            if item.get("name") in virtual_names:
                try:
                    arguments = json.loads(item.get("arguments") or "{}")
                    error = validate_arguments(arguments, ASSISTANT_TOOL["parameters"])
                except (ValueError, TypeError):
                    error = "invalid assistant task arguments"
                if error:
                    state["history"].append(engine._result(item["call_id"], {"error": error}))
                    continue
                if arguments.get("source_id"):
                    engine._finish_retrieval(state, item["call_id"], arguments["source_id"])
                    continue
                state["subtasks"][item["call_id"]] = {
                    "history": [*helper_preamble(state),
                                {"role": "developer", "content": HELPER_ROLE}, {"role": "user", "content": encoded(arguments)}],
                    "steps": 0, "done": False, "status": "active", "request_id": ""}
            else:
                error = ("external tools are unavailable at level 2" if state["level"] == 2 else engine._call_error(state, item))
                if error:
                    state["history"].append(engine._result(item["call_id"], {"error": error}))
                else:
                    emitted.append(engine._queue(state, item, "outer"))
        clear_expert_request(state)
        engine._save(state)
        step_subtasks(engine, state, emitted)
        engine._refresh_phase(state)
        engine._save(state)
        if emitted:
            return emitted
        if engine._has_pending(state):
            return engine._pending_calls(state) or engine._pause(state)
        stop_unfinished_subtasks(engine, state)
        flush_expert_tail(engine, state)
        evaluate_escalation(engine, state)
    return engine._pause(state)


def advance_expert(engine, state, emitted):
    if run_prep(engine, state, emitted):
        return emitted or engine._pending_calls(state) or engine._pause(state)
    step_subtasks(engine, state, emitted)
    engine._refresh_phase(state)
    if emitted:
        return emitted
    if engine._has_pending(state):
        return engine._pending_calls(state) or engine._pause(state)
    stop_unfinished_subtasks(engine, state)
    if state["expert_request"] is None and state["expert_response"] is None:
        flush_expert_tail(engine, state)
        evaluate_escalation(engine, state)
    state["phase"] = "expert_active"
    return outer_expert(engine, state, emitted)
