"""Monotonic dialogue levels and client-executed assistant subdialogues."""
import copy
import hashlib
import json
import math
import os
import threading
import uuid
from pathlib import Path

from coffee_llm.assistant import validate_arguments
from coffee_llm.constants import (
    ASSISTANT_INSTRUCTION,
    ASSISTANT_ROLE,
    ASSISTANT_TOOL,
    ESCALATE_TOOL,
)
from coffee_llm.deflate import deflate_text
from coffee_llm.expert import advance_expert, clear_expert_request
from coffee_llm.limits import COMPACT_EVIDENCE_CHAR_BUDGET, COMPACT_PACKET_CHAR_BUDGET
RESERVED_TOOLS = {"gateway_assistant", "assistant_needs_expert", "gateway_writer", "writer_needs_cloud"}
PAUSE = "Запрос сохранён. Отправьте сообщение ещё раз."


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def bound_text(text, budget, source_id):
    """Оставляет края доказательства и помечает пропуск. Короткий текст не меняется."""
    text = "" if text is None else str(text)
    if budget is None or budget < 1 or len(text) <= budget:
        return text, False
    head, tail = 0, 0
    mark = ""
    for _ in range(3):
        omitted = len(text) - head - tail
        mark = f"\n[пропуск {omitted} символов, source_id={source_id}]\n"
        room = budget - len(mark)
        if room <= 0:
            note = f"[пропуск {len(text)} символов, source_id={source_id}]"
            return (note[:budget] if budget > 0 else note), True
        head = room // 2
        tail = room - head
    return text[:head] + mark + text[len(text) - tail:], True


def trim_edge(text, budget, source_id, edge):
    """edge='end' оставляет начало, edge='start' оставляет конец."""
    text = "" if text is None else str(text)
    if budget is None or budget < 1 or len(text) <= budget:
        return text, False
    label = "отрезано начало" if edge == "start" else "отрезан конец"
    mark = f"\n[{label}, source_id={source_id}]\n"
    room = budget - len(mark)
    if room <= 0:
        note = f"[{label}, source_id={source_id}]"
        return (note[:budget] if budget > 0 else note), True
    if edge == "start":
        return mark + text[-room:], True
    return text[:room] + mark, True


def message_text(item):
    content = item.get("content", "")
    if isinstance(content, str):
        return content
    return "".join(part.get("text") or part.get("refusal") or ""
                   for part in content or [] if isinstance(part, dict))


def is_human(item):
    if item.get("role") != "user":
        return False
    text = message_text(item)
    if "<user_query>" in text:
        return True
    return not any(marker in text for marker in ("<system_reminder>", "<system_notification>", "<user_info>"))


def session_scope(items, explicit=""):
    if explicit:
        return explicit
    prefix = []
    for item in items:
        prefix.append(item)
        if is_human(item):
            break
    return hashlib.sha256(encoded(prefix).encode()).hexdigest()


class DialogueStore:
    def __init__(self, directory):
        self.directory = Path(directory) / "dialogues"
        self.directory.mkdir(parents=True, exist_ok=True)
        self._locks_guard = threading.RLock()
        self._locks = {}

    def lock_for(self, scope):
        with self._locks_guard:
            return self._locks.setdefault(scope, threading.RLock())

    def path(self, scope):
        return self.directory / (hashlib.sha256(scope.encode()).hexdigest() + ".json")

    def load(self, scope):
        path = self.path(scope)
        return json.loads(path.read_text()) if path.exists() else None

    def save(self, scope, state):
        path = self.path(scope)
        temporary = path.with_suffix(".tmp")
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False)
        temporary.replace(path)


class DialogueEngine:
    def __init__(self, store, adjuster, assistant, expert, log=None, max_steps=32,
                 escalation_confidence=0.80, expert_model=None, expert_effort=None,
                 compact_expert_context=False, packet_char_budget=None, evidence_char_budget=None,
                 shrink=None, deflator=None):
        self.store = store
        self.adjuster = adjuster
        self.assistant = assistant
        self.expert = expert
        self.log = log or (lambda *args: None)
        self.max_steps = max_steps
        self.escalation_confidence = escalation_confidence
        self.expert_model = expert_model
        self.expert_effort = expert_effort
        self.compact_expert_context = bool(compact_expert_context)
        self.packet_char_budget = COMPACT_PACKET_CHAR_BUDGET if packet_char_budget is None else packet_char_budget
        self.evidence_char_budget = COMPACT_EVIDENCE_CHAR_BUDGET if evidence_char_budget is None else evidence_char_budget
        self.shrink = shrink
        self.deflator = deflator
        self._context = threading.local()

    def _save(self, state):
        self.store.save(self._context.scope, state)

    def run(self, scope, incoming, tools):
        with self.store.lock_for(scope):
            self._context.scope = scope
            return self._run(scope, copy.deepcopy(incoming), copy.deepcopy(tools))

    def _run(self, scope, incoming, tools):
        fingerprint = hashlib.sha256(encoded([incoming, tools]).encode()).hexdigest()
        state = self._normalize(self.store.load(scope))
        if state and fingerprint in state["deliveries"] and not self._must_resume(state, incoming, state["deliveries"][fingerprint]):
            return copy.deepcopy(state["deliveries"][fingerprint])
        additions = incoming
        if state:
            previous = state["seen"]
            common = 0
            for left, right in zip(previous, incoming):
                if left != right:
                    break
                common += 1
            if common == len(previous):
                additions = incoming[common:]
            elif state["phase"] == "done":
                state = None
            else:
                # Cursor may rewrite earlier context. Keep the server's active history,
                # level and pending calls; reconcile only the client tail.
                additions = self._client_tail(previous, incoming, state)
                self.log("adjuster", "history", "client prefix changed; active dialogue retained")
            if state and state["phase"] == "done" and any(is_human(item) for item in additions):
                state = None
                additions = incoming
        if state is None:
            state = {
                "version": 4, "owner": "undecided", "level": 1, "phase": "initial_decision",
                "history": copy.deepcopy(incoming), "seen": incoming,
                "pending": {}, "subtasks": {}, "deliveries": {}, "tools": tools,
                "initial": {"steps": 0, "guard_required": False, "request_id": ""},
                "expert_deferred": [], "expert_request_id": "", "expert_request": None,
                "expert_response": None, "expert_sent": None, "level_tools": None,
                "escalation_pending": False, "legacy_tools": False,
                "compact": self.compact_expert_context, "archive": [], "prep": None, "expert_ready": False,
            }
        else:
            for item in additions:
                if item.get("type") == "function_call_output":
                    self._receive(state, item)
                elif item.get("role") in ("user", "system", "developer"):
                    busy = (bool(state["expert_deferred"]) or state["expert_request"] is not None
                            or state["expert_response"] is not None or self._has_pending(state)
                            or any(not task["done"] for task in state["subtasks"].values()))
                    if state["owner"] == "expert" and busy:
                        state["expert_deferred"].append(item)
                    else:
                        state["history"].append(item)
                        self._archive_client_item(state, item)
            state["seen"] = incoming
            if state["owner"] != "expert":
                state["tools"] = tools
        self._close_dropped_calls(state, incoming)
        self._save(state)
        result = state.get("final", []) if state["phase"] == "done" else self._advance(state, [])
        if not result:
            result = self._pending_calls(state) or self._pause(state)
        state["seen"] = incoming
        if not self._is_pause(result):
            state["deliveries"][fingerprint] = result
        self._save(state)
        return copy.deepcopy(result)

    @staticmethod
    def _client_tail(previous, incoming, state):
        for old in reversed(previous):
            for index in range(len(incoming) - 1, -1, -1):
                if incoming[index] == old:
                    return incoming[index + 1:]
        known = {encoded(item) for item in previous}
        return [item for item in incoming if (
            item.get("type") == "function_call_output" and item.get("call_id") in state["pending"]
        ) or (is_human(item) and encoded(item) not in known)]

    def _must_resume(self, state, incoming, cached):
        if self._person_is_waiting(state, incoming) and self._open_pending(state):
            return True
        return not cached and self._open_pending(state)

    def _open_pending(self, state):
        return any(not pending.get("done") for pending in state.get("pending", {}).values())

    @staticmethod
    def _is_pause(items):
        return len(items) == 1 and message_text(items[0]) == PAUSE

    def _person_is_waiting(self, state, incoming):
        if any(is_human(item) for item in state.get("expert_deferred") or []):
            return True
        seen = {encoded(item) for item in state.get("seen") or []}
        if any(is_human(item) and encoded(item) not in seen for item in incoming):
            return True
        return self._human_after_open_call(state)

    @staticmethod
    def _human_after_open_call(state):
        for pending in state.get("pending", {}).values():
            if pending.get("done") or pending.get("target") not in ("outer", "initial"):
                continue
            seen_call = False
            for item in state.get("history") or []:
                if item.get("type") == "function_call" and item.get("call_id") == pending.get("original"):
                    seen_call = True
                elif seen_call and is_human(item):
                    return True
        return False

    def _close_dropped_calls(self, state, incoming):
        if not self._person_is_waiting(state, incoming):
            return
        dropped = [(call_id, pending) for call_id, pending in state.get("pending", {}).items() if not pending.get("done")]
        if not dropped:
            return
        for _call_id, pending in dropped:
            pending["done"] = True
            record = self._result(pending["original"], {"error": "client continued without the tool result"})
            self._history_for(state, pending.get("target")).append(record)
            self._archive_client_item(state, record)
            self.log("assistant", "history", "pending tool call closed for a human follow-up")

    def _pending_calls(self, state):
        visible = []
        for call_id, pending in state.get("pending", {}).items():
            if pending.get("done"):
                continue
            history = self._history_for(state, pending.get("target"))
            original = next((item for item in history if item.get("type") == "function_call" and item.get("call_id") == pending.get("original")), None)
            if original is None:
                continue
            item = copy.deepcopy(original)
            item["call_id"] = call_id
            visible.append(item)
        return visible

    def _pause(self, state):
        clear_expert_request(state)
        self.log("expert" if state.get("owner") == "expert" else "assistant", "history", "paused for a human retry")
        self._save(state)
        return [self._answer(PAUSE)]

    def _normalize(self, state):
        if not state:
            return None
        if state.get("version", 1) < 4:
            owner = state.get("owner")
            if owner is None:
                owner = "cloud" if state.get("phase") == "cloud" else (
                    "writer" if state.get("steps", 0) or state.get("pending") else "undecided")
            state["level"] = 3 if owner in ("cloud", "expert") else 1
            state["owner"] = {"cloud": "expert", "writer": "assistant"}.get(owner, owner)
            state["legacy_tools"] = state["level"] == 3
            state["phase"] = {"cloud": "expert_active", "cloud_active": "expert_active",
                              "writer_active": "assistant_active"}.get(state.get("phase"), state.get("phase", "initial_decision"))
            state.setdefault("initial", {"steps": state.pop("steps", 0),
                                         "guard_required": state["owner"] == "assistant", "request_id": ""})
            for suffix, default in (("deferred", []), ("request_id", ""), ("request", None), ("response", None)):
                state["expert_" + suffix] = state.pop("cloud_" + suffix, default)
            state["expert_deferred"].extend(state.pop("cloud_followups", state.pop("deferred", [])))
            state["version"] = 4
        state.setdefault("level", 1)
        state.setdefault("legacy_tools", False)
        state.setdefault("escalation_pending", False)
        state.setdefault("deliveries", {})
        state.setdefault("pending", {})
        state.setdefault("subtasks", {})
        state.setdefault("expert_deferred", [])
        state.setdefault("expert_request_id", "")
        state.setdefault("expert_request", None)
        state.setdefault("expert_response", None)
        state.setdefault("expert_sent", None)
        state.setdefault("level_tools", None)
        state.setdefault("compact", False)
        state.setdefault("archive", [])
        state.setdefault("prep", None)
        state.setdefault("expert_ready", False)
        state["initial"].setdefault("request_id", "")
        for task in state["subtasks"].values():
            task.setdefault("status", "done" if task.get("done") else "active")
            task.setdefault("request_id", "")
            task.pop("guard_required", None)
        if state["owner"] == "assistant":
            for pending in state["pending"].values():
                if pending.get("target") == "outer":
                    pending["target"] = "initial"
        return state

    def _receive(self, state, item):
        pending = state["pending"].get(item.get("call_id"))
        if not pending or pending.get("done"):
            return
        pending["done"] = True
        result = dict(item, call_id=pending["original"])
        self._history_for(state, pending["target"]).append(result)
        self._archive_client_item(state, result)

    def _advance(self, state, emitted):
        if state["owner"] == "undecided":
            decision = self._decision("initial", state["history"], state["tools"], state["level"])
            if decision["action"] == "expert":
                self._select_expert(state)
            else:
                state["owner"], state["phase"] = "assistant", "assistant_active"
        if state["owner"] == "assistant":
            return self._advance_assistant(state, emitted)
        return advance_expert(self, state, emitted)

    def _advance_assistant(self, state, emitted, refreshed=False):
        if self._has_pending(state, "initial"):
            return self._pending_calls(state) or self._pause(state)
        initial = state["initial"]
        if initial["guard_required"]:
            decision = self._decision("assistant", state["history"], state["tools"], state["level"])
            initial["guard_required"] = False
            if decision["action"] == "expert":
                self._select_expert(state)
                return advance_expert(self, state, emitted)
        if initial["steps"] >= self.max_steps:
            if refreshed:
                return self._pause(state)
            # The budget pauses one request. The next client request continues the same history.
            initial["steps"] = 0
            refreshed = True
            self.log("assistant", "budget", "iteration budget refreshed; history kept")
        if not initial["request_id"]:
            initial["request_id"] = "coffee_llm_model_" + uuid.uuid4().hex
        self._save(state)
        self.log("assistant", "request", "purpose=assistant call_id=" + initial["request_id"])
        items = self.assistant([*copy.deepcopy(state["history"]), {"role": "developer", "content": ASSISTANT_ROLE}],
                               self._assistant_tools(state), initial["request_id"])
        initial["steps"] += 1
        state["history"].extend(items)
        initial["request_id"] = ""
        self._save(state)
        calls = [item for item in items if item.get("type") == "function_call"]
        if not calls:
            return self._finish(state, items or [self._answer("Не удалось сформировать ответ.")])
        for item in calls:
            if item.get("name") in ("assistant_needs_expert", "writer_needs_cloud"):
                state["history"].append(self._result(item["call_id"], {
                    "status": "adjustment_requested", "reason": item.get("arguments", "")}))
                continue
            error = self._call_error(state, item)
            if error:
                state["history"].append(self._result(item["call_id"], {"error": error}))
            else:
                emitted.append(self._queue(state, item, "initial", rename=True))
        initial["guard_required"] = True
        return emitted if emitted else self._advance_assistant(state, emitted, refreshed)

    def _assistant_tools(self, state):
        return [*copy.deepcopy([tool for tool in state["tools"] if tool.get("name") not in RESERVED_TOOLS]), copy.deepcopy(ESCALATE_TOOL)]

    def _call_error(self, state, item):
        tool = next((tool for tool in state["tools"] if tool.get("name") == item.get("name") and tool.get("name") not in RESERVED_TOOLS), None)
        if tool is None:
            return "tool is not available: " + str(item.get("name"))
        try:
            arguments = json.loads(item.get("arguments") or "{}")
        except (ValueError, TypeError):
            return "invalid tool arguments JSON"
        return validate_arguments(arguments, tool.get("parameters", {}))

    def _queue(self, state, item, target, rename=False):
        visible = copy.deepcopy(item)
        if rename:
            visible["call_id"] = "coffee_llm_task_" + uuid.uuid4().hex
        state["pending"][visible["call_id"]] = {"target": target, "original": item["call_id"], "done": False}
        return visible

    def _fit_piece(self, text, source_id):
        text = "" if text is None else str(text)
        budget = self.evidence_char_budget
        if budget is None or len(text) <= budget:
            return text
        return deflate_text(
            text, budget, source_id,
            choose=self.shrink, assistant=self.deflator or self.assistant, log=self.log,
        )

    def _history_for(self, state, target):
        if target in ("outer", "initial", None):
            return state["history"]
        if target == "prep":
            prep = state.get("prep") or {}
            return prep.setdefault("history", [])
        task = state.get("subtasks", {}).get(target)
        if task is None:
            return state["history"]
        return task.setdefault("history", [])

    def _archive_client_item(self, state, item):
        if not state.get("compact") or not isinstance(state.get("archive"), list):
            return
        if state.get("expert_ready") or (state.get("prep") or {}).get("status") == "active":
            state["archive"].append(copy.deepcopy(item))

    @staticmethod
    def _call_signature(item):
        raw = item.get("arguments") or "{}"
        try:
            raw = encoded(json.loads(raw))
        except (TypeError, ValueError):
            pass
        return item.get("name", ""), raw

    @staticmethod
    def _outputs_by_signature(history):
        calls = {item.get("call_id"): item for item in history if item.get("type") == "function_call"}
        found = {}
        for item in history:
            if item.get("type") != "function_call_output":
                continue
            call = calls.get(item.get("call_id"))
            if call:
                found[DialogueEngine._call_signature(call)] = item.get("output", "")
        return found

    def _source_histories(self, state):
        histories = [state.get("archive") or [], state.get("history") or []]
        prep = state.get("prep") or {}
        if prep.get("history"):
            histories.append(prep["history"])
        for task in state.get("subtasks", {}).values():
            histories.append(task.get("history") or [])
        return histories

    def _lookup_source(self, state, source_id):
        for history in self._source_histories(state):
            for item in history:
                if item.get("type") == "function_call_output" and str(item.get("call_id")) == str(source_id):
                    return item.get("output", "")
        return None

    def _finish_retrieval(self, state, ident, source_id):
        found = self._lookup_source(state, source_id)
        if found is None:
            result = {
                "status": "stopped",
                "reason": "сохранённый источник не найден",
                "gaps": [source_id],
                "partial_data": {"findings": [], "evidence": [], "actions": [], "limits": [], "gaps": [source_id]},
            }
        else:
            result = {
                "status": "completed",
                "answer": found,
                "evidence": [{"source_id": source_id, "output": found, "omitted": False, "reused": True}],
                "actions": [],
                "limits": [],
                "gaps": [],
                "reused": True,
            }
        state["history"].append(self._result(ident, result))
        if state["level"] == 2:
            state["escalation_pending"] = True

    @staticmethod
    def _bounded_partial(task, budget, fit=None):
        history = task.get("history") or []
        calls = {item.get("call_id"): item for item in history if item.get("type") == "function_call"}
        evidence, actions, limits, gaps, findings = [], [], [], [], []
        for item in history:
            if item.get("type") != "function_call_output":
                continue
            call = calls.get(item.get("call_id"), {})
            source_id = str(item.get("call_id") or "")
            arguments = call.get("arguments", "{}")
            try:
                arguments = json.loads(arguments)
            except (TypeError, ValueError):
                pass
            raw = item.get("output", "")
            if not isinstance(raw, str):
                raw = encoded(raw)
            if fit:
                excerpt = fit(raw, source_id)
                omitted = excerpt != raw
            else:
                excerpt, omitted = bound_text(raw, budget, source_id)
            evidence.append({
                "source_id": source_id, "tool": call.get("name", ""), "arguments": arguments,
                "output": excerpt, "omitted": omitted,
            })
            actions.append({"tool": call.get("name", ""), "arguments": arguments, "source_id": source_id})
            if omitted:
                limits.append("часть доказательства " + source_id + " исключена; запросите source_id через gateway_assistant")
                gaps.append(source_id)
        for item in history:
            if item.get("role") == "assistant" and message_text(item):
                text, omitted = bound_text(message_text(item), budget, "finding")
                findings.append(text)
                if omitted:
                    limits.append("часть текста помощника исключена")
        return {"findings": findings, "evidence": evidence, "actions": actions, "limits": limits, "gaps": gaps}

    def _decision(self, mode, items, tools, level):
        fallback = {"initial": "assistant", "assistant": "continue_assistant",
                    "escalation": "stay_level"}[mode]
        allowed = {"initial": {"assistant", "expert"}, "assistant": {"continue_assistant", "expert"},
                   "escalation": {"stay_level", "expert_tools"}}[mode]
        threshold = None
        try:
            raw = self.adjuster(mode, copy.deepcopy(items), copy.deepcopy(tools))
            if not isinstance(raw, dict) or raw.get("action") not in allowed:
                raise ValueError("invalid adjuster action")
            action = raw["action"]
            confidence = raw.get("confidence", 0.0)
            if action != "expert" and (isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1):
                raise ValueError("invalid adjuster confidence")
            reason = raw.get("reason") or "adjuster selected " + action
            threshold = self.escalation_confidence if action == "expert_tools" else None
            if threshold is not None and confidence < threshold:
                action = fallback
                reason = "escalation confidence below threshold"
        except Exception as error:
            action, confidence, reason = fallback, 0.0, "adjuster failed: " + type(error).__name__
        next_level = 2 if action == "expert" else 3 if action == "expert_tools" else level
        note = f"mode={mode} level={level}->{next_level} decision={action} confidence={confidence}"
        if threshold is not None:
            note += f" threshold={threshold}"
        self.log("adjuster", "decision", note)
        return {"action": action, "confidence": confidence, "reason": reason}

    @staticmethod
    def _has_pending(state, target=None):
        return any(not pending.get("done") and (target is None or pending.get("target") == target) for pending in state["pending"].values())

    @staticmethod
    def _select_expert(state):
        state["owner"], state["phase"], state["level"] = "expert", "expert_active", 2

    @staticmethod
    def _refresh_phase(state):
        if state["phase"] == "done":
            return
        state["phase"] = "helper_active" if any(not task["done"] for task in state["subtasks"].values()) else "expert_active"

    @staticmethod
    def _finish(state, items):
        state["phase"], state["final"] = "done", items
        return items

    @staticmethod
    def _result(ident, value):
        return {"type": "function_call_output", "call_id": ident, "output": encoded(value)}

    @staticmethod
    def _answer(text):
        return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}
