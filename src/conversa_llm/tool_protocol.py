from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

TOOL_START = "<tool_call>"
TOOL_END = "</tool_call>"
_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


@dataclass(frozen=True)
class ParsedToolCall:
    name: str
    arguments: dict[str, Any]


def encode_tool_call(name: str, arguments: dict[str, Any]) -> str:
    payload = json.dumps(
        {"name": name, "arguments": arguments}, ensure_ascii=False, separators=(",", ":")
    )
    return f"{TOOL_START}{payload}{TOOL_END}"


def parse_tool_calls(text: str, allowed_tools: set[str] | None = None) -> list[ParsedToolCall]:
    calls: list[ParsedToolCall] = []
    for raw in _PATTERN.findall(text):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        name = payload.get("name")
        arguments = payload.get("arguments")
        if not isinstance(name, str) or not isinstance(arguments, dict):
            continue
        if allowed_tools is not None and name not in allowed_tools:
            continue
        calls.append(ParsedToolCall(name=name, arguments=arguments))
    return calls


def strip_tool_calls(text: str) -> str:
    return _PATTERN.sub("", text).strip()
