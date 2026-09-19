from conversa_llm.tool_protocol import encode_tool_call, parse_tool_calls, strip_tool_calls


def test_tool_protocol_roundtrip():
    text = encode_tool_call("read", {"path": "src/app.py"})
    calls = parse_tool_calls(text, {"read"})
    assert len(calls) == 1
    assert calls[0].name == "read"
    assert calls[0].arguments == {"path": "src/app.py"}
    assert strip_tool_calls(text) == ""


def test_disallowed_tool_is_ignored():
    text = encode_tool_call("bash", {"command": "pytest -q"})
    assert parse_tool_calls(text, {"read"}) == []
