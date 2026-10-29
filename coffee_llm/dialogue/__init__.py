"""Dialogue engine and its supporting expert/helper modules."""

from coffee_llm.assistant import validate_arguments
from coffee_llm.constants import (
    ASSISTANT_INSTRUCTION,
    ASSISTANT_ROLE,
    ASSISTANT_TOOL,
    ESCALATE_TOOL,
)
from coffee_llm.dialogue.deflate import deflate_text
from coffee_llm.dialogue.engine import (
    PAUSE,
    RESERVED_TOOLS,
    DialogueEngine,
    DialogueStore,
    bound_text,
    encoded,
    is_human,
    message_text,
    session_scope,
    trim_edge,
)
from coffee_llm.dialogue.expert import advance_expert, clear_expert_request
from coffee_llm.limits import COMPACT_EVIDENCE_CHAR_BUDGET, COMPACT_PACKET_CHAR_BUDGET

__all__ = [
    "ASSISTANT_INSTRUCTION",
    "ASSISTANT_ROLE",
    "ASSISTANT_TOOL",
    "COMPACT_EVIDENCE_CHAR_BUDGET",
    "COMPACT_PACKET_CHAR_BUDGET",
    "DialogueEngine",
    "DialogueStore",
    "ESCALATE_TOOL",
    "PAUSE",
    "RESERVED_TOOLS",
    "advance_expert",
    "bound_text",
    "clear_expert_request",
    "deflate_text",
    "encoded",
    "is_human",
    "message_text",
    "session_scope",
    "trim_edge",
    "validate_arguments",
]
