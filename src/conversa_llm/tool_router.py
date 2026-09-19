from __future__ import annotations

import re
from typing import Any

from .tool_protocol import ParsedToolCall

_PATH_RE = re.compile(r"(?P<path>(?:[A-Za-z0-9_.-]+[\\/])*[A-Za-z0-9_.-]+\.[A-Za-z0-9_.-]+)")
_BACKTICK_RE = re.compile(r"`([^`]+)`")


def _available(tools: list[dict[str, Any]]) -> set[str]:
    names = set()
    for tool in tools:
        if isinstance(tool, dict) and tool.get("type") == "function":
            name = (tool.get("function") or {}).get("name")
            if isinstance(name, str):
                names.add(name)
    return names


def _path(text: str) -> str | None:
    match = _PATH_RE.search(text)
    return match.group("path").rstrip(".,;:!?") if match else None


def _command(text: str) -> str | None:
    match = _BACKTICK_RE.search(text)
    if match:
        return match.group(1).strip()
    patterns = [
        r"(?:comando|terminal)\s*[:]?\s*(.+?)(?:\s+agora\.?$|$)",
        r"(?:execute|rode|rodar)\s+(?:no\s+shell\s+|no\s+powershell\s+)?(?:o\s+comando\s+)?(.+?)(?:\s+agora\.?$|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip().strip(".`")
    return None


def route_explicit_tool(prompt: str, tools: list[dict[str, Any]]) -> ParsedToolCall | None:
    """Compila pedidos explícitos e simples para tools do Pi."""
    names = _available(tools)
    lowered = prompt.lower()
    path = _path(prompt)

    if "powershell" in lowered and "powershell" in names and re.search(r"\b(execute|rode|rodar|use)\b", lowered):
        command = _command(prompt)
        if command:
            return ParsedToolCall("powershell", {"command": command})

    if ("bash" in names or "powershell" in names) and (
        " no shell " in f" {lowered} " or " no terminal" in lowered or re.search(r"\b(execute|rode|rodar)\b", lowered)
    ):
        command = _command(prompt)
        if command and not (path and re.search(r"\b(leia|abra|escreva|grave|salve|edite|substitua|troque|altere)\b", lowered)):
            name = "bash" if "bash" in names else "powershell"
            return ParsedToolCall(name, {"command": command})

    if path and "edit" in names and re.search(r"\b(substitua|troque|edite|altere)\b", lowered):
        replacements = [
            r"substitua\s+(.+?)\s+por\s+(.+?)\s+(?:dentro\s+de|no|em)\s+(?:do\s+arquivo\s+)?" + re.escape(path),
            r"troque\s+(.+?)\s+por\s+(.+?)\s+(?:no|em)\s+(?:arquivo\s+)?" + re.escape(path),
            r"altere\s+(?:o\s+texto\s+)?(.+?)\s+para\s+(.+?)\s+(?:no|em)\s+(?:arquivo\s+)?" + re.escape(path),
            r"(?:de|substitua)\s+(.+?)\s+(?:para|por)\s+(.+?)(?:\.|$)",
        ]
        for pattern in replacements:
            match = re.search(pattern, prompt, flags=re.IGNORECASE)
            if match:
                old, new = match.group(1).strip(" `\"'.,"), match.group(2).strip(" `\"'.,")
                if old and new:
                    return ParsedToolCall("edit", {"path": path, "edits": [{"oldText": old, "newText": new}]})

    if path and "write" in names and re.search(r"\b(crie|escreva|grave|salve)\b", lowered):
        content_patterns = [
            r"conte[uú]do\s+exato\s*:\s*(.+)$",
            r"com\s+exatamente\s+este\s+conte[uú]do\s*:\s*(.+)$",
            r"contendo\s*:\s*(.+)$",
            r"(?:escreva\s+em\s+[^:]+|salve\s+o\s+texto\s+.+?\s+no\s+arquivo\s+\S+)\s*:\s*(.+)$",
        ]
        content = None
        for pattern in content_patterns:
            match = re.search(pattern, prompt, flags=re.IGNORECASE | re.DOTALL)
            if match:
                content = match.group(1).strip()
                break
        if content is None:
            match = re.search(r"salve\s+o\s+texto\s+(.+?)\s+no\s+arquivo", prompt, flags=re.IGNORECASE | re.DOTALL)
            if match:
                content = match.group(1).strip(" `\"'")
        if content is not None:
            return ParsedToolCall("write", {"path": path, "content": content})

    if path and "read" in names and re.search(r"\b(leia|abra|mostre|inspecione|confira|analise|veja)\b", lowered):
        return ParsedToolCall("read", {"path": path})

    return None


def choose_tool_call(prompt: str, tools: list[dict[str, Any]], model_calls: list[ParsedToolCall]) -> ParsedToolCall | None:
    explicit = route_explicit_tool(prompt, tools)
    if explicit:
        return explicit
    allowed = _available(tools)
    for call in model_calls:
        if call.name in allowed:
            return call
    return None
