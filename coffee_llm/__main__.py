import json
import math
import os
from pathlib import Path

from coffee_llm.constants import ASSISTANT_CHARACTER, EXPERT_CHARACTER
from coffee_llm.limits import (
    COMPACT_EVIDENCE_CHAR_BUDGET,
    COMPACT_PACKET_CHAR_BUDGET,
    OPENAI_TIMEOUT_SECONDS,
)
from coffee_llm.adjuster import decide, decision_state, model_character
from coffee_llm.openai import assistant_step
from coffee_llm.server import Gateway
from coffee_llm.assistant import local_assistant_turn


_OLD_SETTINGS = {
    "WRITER",
    "WRITER_MODEL",
    "QWEN_MODEL",
    "WRITER_REASONING_EFFORT",
    "DECIDER",
    "NIMBLE_MODEL",
    "JEV_MODEL",
    "OPENAI_MODEL",
    "OPENAI_REASONING_EFFORT",
}


def _file_env(path):
    file = Path(path)
    if not file.exists():
        return {}
    values = {}
    for line in file.read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def _load_settings(path=".env", environ=None):
    file_settings = _file_env(path)
    environment = os.environ if environ is None else environ
    old_keys = sorted((set(file_settings) | set(environment)) & _OLD_SETTINGS)
    if old_keys:
        raise SystemExit("Устаревшие настройки: " + ", ".join(old_keys))
    settings = file_settings
    settings.update({key: value for key, value in environment.items() if value})
    return settings


def _confidence_setting(settings, name, default):
    try:
        confidence = float(settings.get(name, default))
    except (TypeError, ValueError):
        raise SystemExit(name + " должно быть конечным числом от 0 до 1")
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise SystemExit(name + " должно быть конечным числом от 0 до 1")
    return confidence


def _escalation_confidence(settings):
    return _confidence_setting(settings, "ADJUSTER_ESCALATION_CONFIDENCE", "0.80")


def _enabled(settings, name):
    return str(settings.get(name, "0")).lower() in ("1", "true", "yes", "on")


def _char_budget(settings, name, default):
    raw = settings.get(name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise SystemExit(name + " должно быть целым числом больше нуля")
    if value < 1:
        raise SystemExit(name + " должно быть целым числом больше нуля")
    return value


def _or_setting(settings, name, default):
    return settings.get(name) or default


def main():
    settings = _load_settings()
    key = settings.get("OPENAI_API_KEY", "")
    if not key:
        raise SystemExit("В .env нет OPENAI_API_KEY")
    base = settings.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    local_url = settings.get("OLLAMA_URL", "http://127.0.0.1:11434")
    local_assistant = settings.get("ASSISTANT_BACKEND", "openai") == "local"
    assistant_model = (settings.get("ASSISTANT_LOCAL_MODEL", "qwen3.5:9b") if local_assistant
                       else settings.get("ASSISTANT_MODEL", "gpt-6-luna"))
    assistant_effort = settings.get("ASSISTANT_REASONING_EFFORT", "medium")
    expert_model = settings.get("EXPERT_MODEL", "gpt-6.1-sol")
    expert_effort = settings.get("EXPERT_REASONING_EFFORT", "medium")
    escalation_confidence = _escalation_confidence(settings)
    jev = settings.get("ADJUSTER_BACKEND", "local") == "jev"
    adjuster_url = settings.get("JEV_BASE_URL", "https://api.typesafe.ai") if jev else local_url
    adjuster_model = (settings.get("ADJUSTER_JEV_MODEL", "jev-latest") if jev
                      else settings.get("ADJUSTER_LOCAL_MODEL", "nimble-v3:latest"))
    adjuster_key = settings.get("JEV_API_KEY", "") if jev else ""
    if jev and not adjuster_key:
        raise SystemExit("В .env нет JEV_API_KEY")
    assistant_url = local_url if local_assistant else base
    deflate_adjastor_url = _or_setting(settings, "DEFLATE_ADJASTOR_BASE_URL", adjuster_url)
    deflate_adjastor_model = _or_setting(settings, "DEFLATE_ADJASTOR_MODEL", adjuster_model)
    deflator_url = _or_setting(settings, "DEFLATOR_BASE_URL", assistant_url)
    deflator_model = _or_setting(settings, "DEFLATOR_MODEL", assistant_model)

    assistant_character = model_character(
        assistant_model, "локальный вызов" if local_assistant else assistant_effort, ASSISTANT_CHARACTER
    )
    expert_character = model_character(expert_model, expert_effort, EXPERT_CHARACTER)

    def route(mode, items, tools):
        # Keep the whole active task and tool results; do not clip the newest question away.
        url = deflate_adjastor_url if mode == "shrink" else adjuster_url
        model = deflate_adjastor_model if mode == "shrink" else adjuster_model
        state = json.dumps(decision_state(mode, items, tools), ensure_ascii=False)
        return decide(
            url,
            model,
            mode,
            state,
            api_key=adjuster_key,
            assistant=assistant_character,
            expert=expert_character,
        )

    def assistant(messages, tools, request_id=""):
        if local_assistant:
            return local_assistant_turn(
                local_url, assistant_model, messages, tools, OPENAI_TIMEOUT_SECONDS
            )
        return assistant_step(
            base,
            key,
            assistant_model,
            assistant_effort,
            messages,
            tools,
            OPENAI_TIMEOUT_SECONDS,
            request_id,
        )

    def deflator(messages, tools, request_id=""):
        if local_assistant:
            return local_assistant_turn(
                deflator_url, deflator_model, messages, tools, OPENAI_TIMEOUT_SECONDS
            )
        return assistant_step(
            deflator_url, key, deflator_model, assistant_effort,
            messages, tools, OPENAI_TIMEOUT_SECONDS, request_id,
        )

    gateway = Gateway(
        upstream=base,
        api_key=key,
        model=expert_model,
        effort=expert_effort,
        data_path=settings.get("DATA_PATH", "./data"),
        public_model=settings.get("GATEWAY_MODEL", "Coffee LLM"),
        adjuster=route,
        assistant=assistant,
        deflator=deflator,
        escalation_confidence=escalation_confidence,
        compact_expert_context=_enabled(settings, "COMPACT_EXPERT_CONTEXT"),
        packet_char_budget=_char_budget(settings, "COMPACT_PACKET_CHAR_BUDGET", COMPACT_PACKET_CHAR_BUDGET),
        evidence_char_budget=_char_budget(settings, "COMPACT_EVIDENCE_CHAR_BUDGET", COMPACT_EVIDENCE_CHAR_BUDGET),
        proxy=settings.get("PROXY", "0").lower() in ("1", "true", "yes", "on"),
        port=int(settings.get("PORT", "8080")),
        metrics_compare=_enabled(settings, "METRICS_COMPARE"),
        metrics_repository=settings.get("METRICS_REPOSITORY", ""),
    )
    if gateway.metrics_compare and not gateway.metrics_repository:
        raise SystemExit("В .env нет METRICS_REPOSITORY")
    print("adjuster " + adjuster_model, flush=True)
    print("assistant " + assistant_model + " " + assistant_effort, flush=True)
    print("expert " + expert_model + " " + expert_effort, flush=True)
    print("deflate_adjastor " + deflate_adjastor_model, flush=True)
    print("deflator " + deflator_model, flush=True)
    print("compact expert context " + ("on" if gateway.compact_expert_context else "off"), flush=True)
    print("metrics compare " + ("on " + gateway.metrics_repository if gateway.metrics_compare else "off"), flush=True)
    gateway.start()
    print("coffee_llm listening " + gateway.url, flush=True)
    gateway.thread.join()


if __name__ == "__main__":
    main()
