from __future__ import annotations

from typing import Any

from .agent_prompt import _content_text


def render_decision_state(messages: list[dict[str, Any]]) -> str:
    """Estado compacto para a camada System-One.

    Não inclui schemas de tools: o espaço de ações é fixo e tipado.
    """
    rows: list[str] = []
    for message in messages[-12:]:
        role = message.get("role")
        if role in {"system", "developer"}:
            continue
        text = _content_text(message.get("content")).strip()
        if role == "user" and text:
            rows.append("USER: " + text)
        elif role == "assistant":
            if text:
                rows.append("ASSISTANT: " + text)
            for call in message.get("tool_calls") or []:
                fn = (call or {}).get("function") or {}
                name = fn.get("name")
                if isinstance(name, str):
                    rows.append("LAST_TOOL_CALL: " + name)
        elif role == "tool":
            name = message.get("name") or "tool"
            rows.append(f"TOOL_RESULT {name}: {text[:1200]}")
    return "\n".join(rows)[-12000:]
