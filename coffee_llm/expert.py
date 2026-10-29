"""Compatibility exports for the dialogue expert implementation."""

from coffee_llm.dialogue.expert import (
    _assistant_task,
    _next_slice,
    advance_expert,
    clear_expert_request,
    evaluate_escalation,
    expert_tools,
    flush_expert_tail,
    outer_expert,
    stable_expert_history,
)

__all__ = [
    "_assistant_task",
    "_next_slice",
    "advance_expert",
    "clear_expert_request",
    "evaluate_escalation",
    "expert_tools",
    "flush_expert_tail",
    "outer_expert",
    "stable_expert_history",
]
