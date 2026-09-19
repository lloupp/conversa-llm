from __future__ import annotations

PI_CORE_TOOLS = [
    {"type": "function", "function": {"name": "read", "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "offset": {"type": "integer"},
            "limit": {"type": "integer"},
        },
        "required": ["path"],
    }}},
    {"type": "function", "function": {"name": "write", "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"],
    }}},
    {"type": "function", "function": {"name": "edit", "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string"}, "edits": {"type": "array"}},
        "required": ["path", "edits"],
    }}},
    {"type": "function", "function": {"name": "bash", "parameters": {
        "type": "object",
        "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}},
        "required": ["command"],
    }}},
    {"type": "function", "function": {"name": "powershell", "parameters": {
        "type": "object",
        "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}},
        "required": ["command"],
    }}},
    {"type": "function", "function": {"name": "grep", "parameters": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string"},
            "glob": {"type": "string"},
            "ignoreCase": {"type": "boolean"},
            "literal": {"type": "boolean"},
            "context": {"type": "number"},
            "limit": {"type": "number"},
        },
        "required": ["pattern"],
    }}},
    {"type": "function", "function": {"name": "find", "parameters": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string"},
            "limit": {"type": "number"},
        },
        "required": ["pattern"],
    }}},
    {"type": "function", "function": {"name": "ls", "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string"}, "limit": {"type": "number"}},
        "required": [],
    }}},
]
