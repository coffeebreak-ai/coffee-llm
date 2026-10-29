import hashlib
import json
import math
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

_CALL_ID_OK = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def fit_call_id(value):
    text = str(value or "")
    if _CALL_ID_OK.fullmatch(text):
        return text
    if not text:
        return ""
    return hashlib.sha256(text.encode()).hexdigest()


def upstream_url(base, path):
    root = base.rstrip("/")
    if root.endswith("/v1") and (path == "/v1" or path.startswith("/v1/")):
        path = path[3:] or "/"
    if not path.startswith("/"):
        path = "/" + path
    return root + path


CHAT_DROPPED = ("temperature", "top_p", "logprobs", "top_logprobs")
RESPONSES_DROPPED = ("temperature", "top_p")


def chat_needs_responses(payload):
    if payload.get("tools"):
        return True
    for message in payload.get("messages") or []:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "tool" or message.get("tool_calls"):
            return True
    return False


def distinct_items(items):
    seen = set()
    unique = []
    for item in items or []:
        ident = item.get("id") if isinstance(item, dict) else None
        if ident and ident in seen:
            continue
        if ident:
            seen.add(ident)
        unique.append(item)
    return unique


def chat_input(messages):
    items = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        if "_response_item" in message:
            items.append(message["_response_item"])
            continue
        role = message.get("role")
        if role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": fit_call_id(message.get("tool_call_id")),
                    "output": _message_text(message.get("content")),
                }
            )
            continue
        text = _message_text(message.get("content"))
        if text and role in ("system", "user", "assistant", "developer"):
            items.append({"role": role, "content": text})
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function") or {}
            items.append(
                {
                    "type": "function_call",
                    "call_id": fit_call_id(call.get("id")),
                    "name": function.get("name") or "",
                    "arguments": function.get("arguments") or "{}",
                }
            )
    return items


def chat_tools(tools):
    converted = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        if tool.get("type") not in (None, "function"):
            converted.append(tool)
            continue
        item = {"type": "function", "name": function.get("name") or ""}
        if function.get("description"):
            item["description"] = function["description"]
        if function.get("parameters"):
            item["parameters"] = function["parameters"]
        converted.append(item)
    return converted


def chat_to_responses(payload, effort):
    tools = chat_tools(payload.get("tools"))
    body = {
        "model": payload.get("model"),
        "input": chat_input(payload.get("messages")),
        "reasoning": {"effort": effort},
        "stream": False,
    }
    if tools:
        body["tools"] = tools
    return body


def responses_as_chat(data, stream):
    text_parts = []
    tool_calls = []
    for item in data.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function_call":
            tool_calls.append(
                {
                    "id": item.get("call_id") or "",
                    "type": "function",
                    "function": {
                        "name": item.get("name") or "",
                        "arguments": item.get("arguments") or "",
                    },
                }
            )
            continue
        if item.get("type") != "message":
            continue
        for chunk in item.get("content") or []:
            if isinstance(chunk, dict):
                text_parts.append(chunk.get("text") or "")
            elif isinstance(chunk, str):
                text_parts.append(chunk)
    text = "".join(text_parts)
    message = {"role": "assistant", "content": text or None}
    finish = "stop"
    if tool_calls:
        message["tool_calls"] = tool_calls
        finish = "tool_calls"
    completion_id = data.get("id") or "chatcmpl-coffee_llm"
    if not stream:
        payload = {
            "id": completion_id,
            "object": "chat.completion",
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        }
        return "application/json", json.dumps(payload, ensure_ascii=False).encode()
    chunks = []
    delta = {"role": "assistant"}
    if text:
        delta["content"] = text
    chunks.append(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
    )
    for index, call in enumerate(tool_calls):
        chunks.append(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": index,
                                    "id": call["id"],
                                    "type": "function",
                                    "function": call["function"],
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            }
        )
    chunks.append(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
        }
    )
    lines = ["data: " + json.dumps(chunk, ensure_ascii=False) for chunk in chunks]
    lines.append("data: [DONE]")
    lines.append("")
    return "text/event-stream", ("\n\n".join(lines) + "\n").encode()


def _message_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or ""))
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


RATE_LIMIT_RETRY_FALLBACK_SECONDS = 60
_RATE_LIMIT_CODES = {"rate_limit_exceeded", "too_many_requests", "slow_down"}
_QUOTA_CODES = {
    "insufficient_quota", "quota_exceeded", "billing_hard_limit_reached",
    "billing_not_active", "spend_limit_exceeded",
}
_RESET_PART = re.compile(r"(\d+(?:\.\d+)?)(ms|s|m|h)")


def _retry_after_seconds(value):
    if not value:
        return None
    try:
        seconds = float(value)
        return seconds if math.isfinite(seconds) and seconds >= 0 else None
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def _reset_seconds(value):
    if not value:
        return None
    text = str(value).strip()
    parts = list(_RESET_PART.finditer(text))
    if not parts or "".join(part.group(0) for part in parts) != text:
        return None
    units = {"ms": 0.001, "s": 1, "m": 60, "h": 3600}
    seconds = sum(float(part.group(1)) * units[part.group(2)] for part in parts)
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def _rate_limit_retry_seconds(error):
    if error.code != 429:
        return None
    try:
        raw = error.read()
    finally:
        error.close()
    try:
        payload = json.loads(raw)
        details = payload.get("error", {}) if isinstance(payload, dict) else {}
    except (TypeError, ValueError):
        details = {}
    if not isinstance(details, dict):
        details = {}
    code = str(details.get("code") or "").lower()
    error_type = str(details.get("type") or "").lower()
    message = str(details.get("message") or "").lower()
    if code in _QUOTA_CODES or error_type in _QUOTA_CODES or any(
        phrase in message for phrase in ("insufficient quota", "current quota", "billing", "spend limit", "out of credits")
    ):
        return None

    headers = error.headers or {}
    retry_after = _retry_after_seconds(headers.get("Retry-After"))
    resets = [
        _reset_seconds(headers.get("x-ratelimit-reset-tokens")),
        _reset_seconds(headers.get("x-ratelimit-reset-requests")),
        _reset_seconds(headers.get("x-ratelimit-reset-project-tokens")),
    ]
    resets = [value for value in resets if value is not None]
    is_rate_limit = (
        code in _RATE_LIMIT_CODES or error_type in {"rate_limit_error", "rate_limit_exceeded"}
        or "rate limit" in message or "too many requests" in message or retry_after is not None or resets
    )
    if not is_rate_limit:
        return None

    minimum_wait = max([value for value in [retry_after, *resets] if value is not None], default=RATE_LIMIT_RETRY_FALLBACK_SECONDS)
    error.gateway_retry_after = math.ceil(minimum_wait)
    return minimum_wait


def usage_record(payload):
    """Поля usage, которые поставщик действительно прислал. Отсутствие — не ноль."""
    usage = payload.get("usage") if isinstance(payload, dict) else None
    if not isinstance(usage, dict):
        return {}
    record = {}
    for key in ("input_tokens", "output_tokens"):
        if usage.get(key) is not None:
            record[key] = usage[key]
    details = usage.get("input_tokens_details")
    if isinstance(details, dict) and details.get("cached_tokens") is not None:
        record["cached_tokens"] = details["cached_tokens"]
    output_details = usage.get("output_tokens_details")
    if isinstance(output_details, dict) and output_details.get("reasoning_tokens") is not None:
        record["reasoning_tokens"] = output_details["reasoning_tokens"]
    return record


def usage_note(record, *, call_id="", duration_ms=None):
    parts = []
    if call_id:
        parts.append("call_id=" + str(call_id))
    if duration_ms is not None:
        parts.append("duration_ms=" + str(duration_ms))
    for key in ("input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens"):
        if isinstance(record, dict) and key in record:
            parts.append(f"{key}={record[key]}")
    cached = record.get("cached_tokens") if isinstance(record, dict) else None
    if cached is not None:
        try:
            hit = int(cached) > 0
        except (TypeError, ValueError):
            hit = False
        parts.append("cache_hit=" + ("1" if hit else "0"))
    return " ".join(parts)


def responses_request(base, api_key, body, timeout, request_id="", *, retry_rate_limits=False):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if request_id:
        headers["Idempotency-Key"] = request_id
    request = urllib.request.Request(
        upstream_url(base, "/v1/responses"), data=json.dumps(body).encode(),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        if retry_rate_limits:
            _rate_limit_retry_seconds(error)
        raise


def assistant_step(base, api_key, model, effort, messages, tools, timeout, request_id=""):
    body = chat_to_responses({"model": model, "messages": messages, "tools": tools}, effort)
    data = responses_request(base, api_key, body, timeout, request_id)
    _, rendered = responses_as_chat(data, False)
    message = json.loads(rendered)["choices"][0]["message"]
    message["_response_items"] = data.get("output", [])
    message["_usage"] = usage_record(data)
    return message


def chat_text(base, api_key, model, effort, system, user, timeout):
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "reasoning_effort": effort,
    }
    if effort != "none":
        for key in CHAT_DROPPED:
            body.pop(key, None)
    request = urllib.request.Request(
        upstream_url(base, "/v1/chat/completions"),
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.load(response)
    message = data["choices"][0]["message"]
    return _message_text(message.get("content")).strip()


def prepare_expert_body(payload, *, endpoint, model, effort):
    body = dict(payload)
    body["model"] = model
    if endpoint == "responses":
        reasoning = dict(body.get("reasoning") or {})
        reasoning["effort"] = effort
        body["reasoning"] = reasoning
        for key in RESPONSES_DROPPED:
            body.pop(key, None)
        return body
    if effort is not None:
        body["reasoning_effort"] = effort
    for key in CHAT_DROPPED:
        body.pop(key, None)
    return body
