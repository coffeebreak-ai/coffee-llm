"""OpenAI-compatible HTTP interface; external tools remain client-executed."""
import copy
import json
import threading
import time
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from coffee_llm.constants import ASSISTANT_INSTRUCTION
from coffee_llm.dialogue import DialogueEngine, DialogueStore, session_scope
from coffee_llm.metrics import continue_comparison
from coffee_llm.limits import MAX_GATEWAY_TURNS, MAX_REQUEST_BYTES, OPENAI_TIMEOUT_SECONDS
from coffee_llm.openai import (
    chat_input, chat_tools, prepare_expert_body, responses_as_chat, responses_request,
    usage_note, usage_record,
)
from coffee_llm.traffic import TrafficLog

LOG_ROLES = {
    "adjuster": "adjuster",
    "assistant": "assistant",
    "expert": "expert",
    "decider": "adjuster",
    "writer": "assistant",
    "cloud": "expert",
}


def log_role(who):
    return LOG_ROLES.get(who, who)


def response_items(payload, endpoint):
    if endpoint == "chat":
        return chat_input(payload.get("messages"))
    items = payload.get("input", [])
    if isinstance(items, str):
        items = [{"role": "user", "content": items}]
    items = copy.deepcopy(items)
    if payload.get("instructions"):
        items = [{"role": "system", "content": payload["instructions"]}, *items]
    return items


def response_tools(payload, endpoint):
    if endpoint == "chat":
        tools = chat_tools(payload.get("tools"))
    else:
        tools = payload.get("tools", [])
    return [copy.deepcopy(t) for t in tools if t.get("name") not in ("gateway_assistant", "assistant_needs_expert")]


def chat_messages(items):
    messages = []
    for item in items:
        kind = item.get("type")
        if kind == "function_call":
            messages.append({"role": "assistant", "content": None, "tool_calls": [{
                "id": item["call_id"], "type": "function", "function": {
                    "name": item["name"], "arguments": item.get("arguments", "{}")}}]})
        elif kind == "function_call_output":
            messages.append({"role": "tool", "tool_call_id": item["call_id"], "content": item.get("output", "")})
        elif item.get("role"):
            messages.append({"role": item["role"], "content": item.get("content", "")})
        else:
            messages.append({"_response_item": copy.deepcopy(item)})
    return messages


def assistant_items(message):
    if "_response_items" in message:
        return message["_response_items"]
    items = chat_input([message])
    for item in items:
        if item.get("role") == "assistant":
            item["type"] = "message"
            item["content"] = [{"type": "output_text", "text": item["content"]}]
    return items


def render_response(items, endpoint, stream):
    payload = {"object": "response", "status": "completed", "output": items}
    if endpoint == "chat":
        return responses_as_chat(payload, stream)
    if not stream:
        return "application/json", json.dumps(payload, ensure_ascii=False).encode()
    events = []
    for index, item in enumerate(items):
        events.append({"type": "response.output_item.added", "output_index": index, "item": item})
        events.append({"type": "response.output_item.done", "output_index": index, "item": item})
    events.append({"type": "response.completed", "response": payload})
    return "text/event-stream", "".join(
        "event: " + event["type"] + "\ndata: " + json.dumps(event, ensure_ascii=False) + "\n\n"
        for event in events
    ).encode()


class Gateway:
    def __init__(self, *, upstream, api_key, model, effort, data_path,
                 public_model="Coffee LLM", adjuster=None, assistant=None, proxy=False, port=0,
                 max_steps=MAX_GATEWAY_TURNS, escalation_confidence=0.80,
                 compact_expert_context=False, packet_char_budget=None, evidence_char_budget=None,
                 deflator=None, metrics_compare=False, metrics_repository=""):
        self.upstream = upstream
        self.api_key = api_key
        self.model = model
        self.effort = effort
        self.public_model = public_model
        self.adjuster = adjuster or (
            lambda mode, items, tools: {
                "action": {"initial": "expert", "escalation": "stay_level"}[mode],
                "reason": "adjuster is not configured",
            }
        )
        self.assistant = assistant
        self.deflator = deflator
        self.escalation_confidence = escalation_confidence
        self.proxy = proxy
        self.port = port
        self.max_steps = max_steps
        self.compact_expert_context = compact_expert_context
        self.packet_char_budget = packet_char_budget
        self.evidence_char_budget = evidence_char_budget
        self.metrics_compare = metrics_compare
        self.metrics_repository = metrics_repository or ""
        self._usage_local = threading.local()
        self.last_metrics = None
        self.traffic = TrafficLog(data_path)
        self.store = DialogueStore(data_path)
        self.httpd = None
        self.thread = None

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def start(self):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", self.port), request_handler(self))
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def stop(self):
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()

    def traffic_text(self):
        return self.traffic.text()


def request_handler(gateway):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, content_type, body):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status, message, headers=None):
            body = json.dumps({"error": {"message": message}}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            for name, value in (headers or {}).items():
                self.send_header(name, str(value))
            self.end_headers()
            self.wfile.write(body)

        def _model_log(self, who, status, note=""):
            gateway.traffic.access("POST", "/v1/responses", status, note, log_role(who))

        def _usage_log(self, who, request_id, started, record):
            duration_ms = int((time.perf_counter() - started) * 1000)
            self._model_log(who, "usage", usage_note(record or {}, call_id=request_id, duration_ms=duration_ms))
            bucket = getattr(gateway._usage_local, "bucket", None)
            if bucket is not None and record:
                bucket.append((log_role(who), record))

        def _collect(self, fn):
            bucket = []
            gateway._usage_local.bucket = bucket
            try:
                return fn(), list(bucket)
            finally:
                gateway._usage_local.bucket = None

        def _expert(self, items, tools, request_id="", model=None, effort=None):
            request_model = model or gateway.model
            request_effort = effort or gateway.effort
            body = prepare_expert_body({"input": items, "tools": tools, "stream": False},
                                       endpoint="responses", model=request_model, effort=request_effort)
            started = time.perf_counter()
            try:
                data = responses_request(
                    gateway.upstream,
                    gateway.api_key,
                    body,
                    OPENAI_TIMEOUT_SECONDS,
                    request_id,
                    retry_rate_limits=True,
                )
            except Exception as error:
                status = f"HTTP {error.code}" if isinstance(error, urllib.error.HTTPError) else type(error).__name__
                self._model_log("expert", "error", status)
                raise
            self._model_log("expert", 200)
            self._usage_log("expert", request_id, started, usage_record(data))
            return {
                "status": data.get("status", "completed"),
                "output": data.get("output", []),
            }

        def _assistant(self, items, tools, request_id=""):
            if gateway.assistant is None:
                return [{"type": "function_call", "name": "assistant_needs_expert", "call_id": "assistant_unavailable",
                         "arguments": '{"reason":"assistant is not configured"}'}]
            messages = [{"role": "system", "content": ASSISTANT_INSTRUCTION}, *chat_messages(items)]
            requested_tools = [{"type": "function", "function": {k: v for k, v in t.items() if k != "type"}}
                               for t in tools]
            started = time.perf_counter()
            try:
                result = gateway.assistant(messages, requested_tools, request_id)
            except Exception as error:
                self._model_log("assistant", "error", type(error).__name__)
                raise
            self._model_log("assistant", 200)
            record = result.get("_usage") if isinstance(result, dict) else {}
            self._usage_log("assistant", request_id, started, record if isinstance(record, dict) else {})
            return assistant_items(result)

        def _adjuster(self, mode, items, tools):
            self._model_log("adjuster", "request", mode)
            try:
                result = gateway.adjuster(mode, items, tools)
                if isinstance(result, str):
                    result = {"action": result, "reason": ""}
                allowed = {
                    "initial": {"assistant", "expert"},
                    "assistant": {"continue_assistant", "expert"},
                    "escalation": {"stay_level", "expert_tools"},
                    "shrink": {"cut_start", "cut_end", "compress"},
                }.get(mode)
                if allowed is None or result.get("action") not in allowed:
                    raise ValueError("invalid adjuster action")
            except Exception as error:
                self._model_log("adjuster", "error", type(error).__name__)
                raise
            self._model_log("adjuster", 200, "decision=" + result["action"])
            return result

        def _metrics_session(self, scope, items, tools, engine):
            def collected_run(run_scope, run_items, run_tools):
                return self._collect(lambda: engine.run(run_scope, run_items, run_tools))

            def expert_turn(history, run_tools, request_id):
                return self._collect(lambda: self._expert(history, run_tools, request_id=request_id))

            def store_report(report):
                gateway.last_metrics = report

            return continue_comparison(
                gateway.store.directory.parent, scope, items, tools,
                repository=gateway.metrics_repository,
                plain_run=engine.run,
                collected_run=collected_run,
                expert_turn=expert_turn,
                log_note=lambda note: self._model_log("expert", "metrics", note),
                store_report=store_report,
            )

        def _shrink(self, text, source_id):
            result = gateway.adjuster("shrink", [{"role": "user", "content": text}], [{"source_id": source_id}])
            if isinstance(result, str):
                return result
            return (result or {}).get("action")

        def do_GET(self):
            if self.path.split("?", 1)[0] != "/v1/models":
                self._error(404, "Not found")
                return
            self._send(200, "application/json", json.dumps({"object": "list", "data": [
                {"id": gateway.public_model, "object": "model", "owned_by": "coffee_llm"}]}).encode())

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            if path not in ("/v1/responses", "/v1/chat/completions", "/responses", "/chat/completions"):
                self._error(404, "Not found")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 0 or length > MAX_REQUEST_BYTES:
                    self._error(413, "Request too large")
                    return
                raw = self.rfile.read(length)
                payload = json.loads(raw)
                endpoint = "responses" if path.endswith("responses") else "chat"
                items = response_items(payload, endpoint)
                tools = response_tools(payload, endpoint)
                def log(who, event, note):
                    self._model_log(who, event, note)
                if gateway.proxy:
                    self._model_log("expert", "request", "proxy")
                    result = self._expert(items, tools)["output"]
                else:
                    scope = session_scope(items, self.headers.get("X-Coffee-Llm-Session", ""))
                    engine = DialogueEngine(gateway.store, self._adjuster, self._assistant, self._expert,
                                            log=log, max_steps=gateway.max_steps,
                                            escalation_confidence=gateway.escalation_confidence,
                                            expert_model=gateway.model, expert_effort=gateway.effort,
                                            compact_expert_context=gateway.compact_expert_context,
                                            packet_char_budget=gateway.packet_char_budget,
                                            evidence_char_budget=gateway.evidence_char_budget,
                                            shrink=self._shrink, deflator=gateway.deflator)
                    if gateway.metrics_compare:
                        result = self._metrics_session(scope, items, tools, engine)
                    else:
                        result = engine.run(scope, items, tools)
                content_type, body = render_response(result, endpoint, bool(payload.get("stream")))
                self._send(200, content_type, body)
                gateway.traffic.write(raw.decode(errors="replace"), body.decode(errors="replace"))
            except (ValueError, TypeError, KeyError) as error:
                self._error(400, str(error))
            except urllib.error.HTTPError as error:
                headers = {}
                retry_after = getattr(error, "gateway_retry_after", None)
                if retry_after is not None:
                    headers["Retry-After"] = retry_after
                message = "Model provider rejected the request"
                if retry_after is not None and not gateway.proxy:
                    message = "Лимит частоты поставщика. Запрос сохранён, отправьте сообщение ещё раз."
                self._error(error.code, message, headers)
            except (OSError, RuntimeError, TimeoutError) as error:
                self._error(502, "Model request failed: " + type(error).__name__)

        def log_message(self, format, *args):
            return
    return Handler
