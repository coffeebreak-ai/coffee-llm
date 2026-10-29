"""Сжатие длинного ответа помощника перед экспертом.

Роли этого шага отдельные: решатель deflate_adjastor и помощник deflator.
Сейчас им можно указать те же адреса, что у аджастера и ассистента.
"""

DEFLATE_ADJASTOR = "deflate_adjastor"
DEFLATOR = "deflator"


def _assistant_text(items):
    from coffee_llm.dialogue import message_text

    if isinstance(items, str):
        return items.strip()
    if isinstance(items, dict):
        nested = items.get("_response_items")
        if isinstance(nested, list):
            items = nested
        else:
            return message_text(items).strip()
    if not isinstance(items, list):
        return ""
    parts = []
    for item in items:
        if isinstance(item, str):
            parts.append(item)
            continue
        if isinstance(item, dict) and (item.get("role") == "assistant" or item.get("type") == "message"):
            parts.append(message_text(item))
    return "\n".join(part for part in parts if part).strip()


def deflate_text(text, budget, source_id, *, choose, assistant, log):
    from coffee_llm.constants import SHRINK_ROLE
    from coffee_llm.dialogue import bound_text, trim_edge

    text = "" if text is None else str(text)
    if budget is None or len(text) <= budget or choose is None:
        return bound_text(text, budget, source_id)[0] if budget and len(text) > budget else text
    log(DEFLATE_ADJASTOR, "request", f"source_id={source_id} chars={len(text)}")
    try:
        action = choose(text, source_id)
    except Exception as error:
        log(DEFLATE_ADJASTOR, "error", type(error).__name__)
        return bound_text(text, budget, source_id)[0]
    if action not in ("cut_start", "cut_end", "compress"):
        log(DEFLATE_ADJASTOR, "error", "invalid decision")
        return bound_text(text, budget, source_id)[0]
    log(DEFLATE_ADJASTOR, 200, "decision=" + action)
    if action == "cut_start":
        return trim_edge(text, budget, source_id, "start")[0]
    if action == "cut_end":
        return trim_edge(text, budget, source_id, "end")[0]
    request_id = "coffee_llm_deflate_" + str(source_id)
    log(DEFLATOR, "request", f"source_id={source_id} chars={len(text)} call_id={request_id}")
    try:
        items = assistant([
            {"role": "developer", "content": SHRINK_ROLE},
            {"role": "user", "content": "source_id=" + str(source_id) + "\n" + text},
        ], [], request_id)
        summary = _assistant_text(items)
    except Exception as error:
        log(DEFLATOR, "error", type(error).__name__)
        return bound_text(text, budget, source_id)[0]
    log(DEFLATOR, 200, f"chars={len(summary)}")
    if not summary:
        return bound_text(text, budget, source_id)[0]
    if len(summary) <= budget:
        return summary
    return bound_text(summary, budget, source_id)[0]
