"""Structured assistant turns and validation of tool arguments."""
import json
import urllib.request
import uuid


def validate_arguments(value, schema):
    """Return a validation error, or an empty string. Never coerce argument types."""
    if not isinstance(schema, dict):
        return "invalid tool schema"
    for keyword in ("anyOf", "oneOf"):
        if keyword in schema:
            matches = sum(not validate_arguments(value, candidate) for candidate in schema[keyword])
            if not matches or keyword == "oneOf" and matches != 1:
                return f"arguments do not match {keyword}"
    kind = schema.get("type")
    types = {
        "object": lambda v: isinstance(v, dict),
        "array": lambda v: isinstance(v, list),
        "string": lambda v: isinstance(v, str),
        "number": lambda v: isinstance(v, (float, int)) and not isinstance(v, bool),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "boolean": lambda v: isinstance(v, bool),
        "null": lambda v: v is None,
    }
    kinds = kind if isinstance(kind, list) else [kind] if kind else []
    if kinds and not any(types.get(k, lambda v: False)(value) for k in kinds):
        return f"expected {kind}"
    if "enum" in schema and value not in schema["enum"]:
        return "value is outside enum"
    if isinstance(value, dict):
        for name in schema.get("required", []):
            if name not in value:
                return f"missing required argument: {name}"
        properties = schema.get("properties", {})
        for name, item in value.items():
            if name not in properties and schema.get("additionalProperties") is False:
                return f"unknown argument: {name}"
            child = properties.get(name, schema.get("additionalProperties", {}))
            if isinstance(child, dict):
                error = validate_arguments(item, child)
                if error:
                    return f"{name}: {error}"
    if isinstance(value, list):
        for item in value:
            error = validate_arguments(item, schema.get("items", {}))
            if error:
                return error
    return ""


def local_assistant_turn(url, model, messages, tools, timeout):
    body = {"model": model, "messages": messages, "tools": tools, "stream": False}
    request = urllib.request.Request(url.rstrip("/") + "/api/chat",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        message = json.load(response)["message"]
    for item in message.get("tool_calls") or []:
        item.setdefault("id", "assistant_" + uuid.uuid4().hex)
        function = item.setdefault("function", {})
        if isinstance(function.get("arguments"), dict):
            function["arguments"] = json.dumps(function["arguments"], ensure_ascii=False)
    return message
