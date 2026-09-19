from __future__ import annotations

import json
from typing import Any

from .tool_protocol import encode_tool_call


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in {"text", "input_text"}:
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def compact_tool_catalog(tools: list[dict[str, Any]]) -> str:
    rows: list[str] = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        fn = tool.get("function") or {}
        name = fn.get("name")
        if not isinstance(name, str):
            continue
        parameters = fn.get("parameters") or {}
        properties = parameters.get("properties") or {}
        required = set(parameters.get("required") or [])
        fields = []
        for field, schema in properties.items():
            if not isinstance(schema, dict):
                schema = {}
            typ = schema.get("type", "any")
            suffix = "" if field in required else "?"
            fields.append(f"{field}{suffix}:{typ}")
        rows.append(f"- {name}({', '.join(fields)})")
    return "\n".join(rows)


def render_agent_prompt(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
    sections = [
        "Você é um agente de código. Responda em texto quando não precisar de ferramenta.",
        "Quando precisar usar ferramenta, responda SOMENTE com <tool_call>{JSON}</tool_call>.",
    ]
    catalog = compact_tool_catalog(tools)
    if catalog:
        sections.append("FERRAMENTAS DISPONÍVEIS:\n" + catalog)

    history: list[str] = []
    for message in messages:
        role = message.get("role")
        if role in {"system", "developer"}:
            continue
        if role == "user":
            text = _content_text(message.get("content")).strip()
            if text:
                history.append("USUÁRIO:\n" + text)
        elif role == "assistant":
            text = _content_text(message.get("content")).strip()
            if text:
                history.append("ASSISTENTE:\n" + text)
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function") or {}
                name = fn.get("name")
                raw_args = fn.get("arguments", "{}")
                if isinstance(raw_args, str):
                    try:
                        args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        args = {}
                elif isinstance(raw_args, dict):
                    args = raw_args
                else:
                    args = {}
                if isinstance(name, str):
                    history.append("ASSISTENTE:\n" + encode_tool_call(name, args))
        elif role == "tool":
            text = _content_text(message.get("content")).strip()
            tool_name = message.get("name") or "tool"
            call_id = message.get("tool_call_id") or ""
            history.append(f"RESULTADO {tool_name} {call_id}:\n{text}")

    sections.append("HISTÓRICO:\n" + "\n\n".join(history[-10:]))
    sections.append("ASSISTENTE:")
    return "\n\n".join(sections)


def fit_prompt(tokenizer, prompt: str, context_length: int, reserve_tokens: int) -> list[int]:
    prefix = tokenizer.encode_prompt(prompt)
    budget = max(8, context_length - max(1, reserve_tokens))
    if len(prefix) <= budget:
        return prefix
    head = prefix[:2]
    tail = prefix[-max(1, budget - len(head)) :]
    return head + tail
