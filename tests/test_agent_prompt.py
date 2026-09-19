import json

from conversa_llm.agent_prompt import render_agent_prompt
from conversa_llm.pi_tools import PI_CORE_TOOLS as TOOLS


def test_prompt_keeps_tool_result_and_omits_huge_system():
    messages = [
        {"role": "system", "content": "X" * 10000},
        {"role": "user", "content": "Leia README.md"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "c1",
                "type": "function",
                "function": {
                    "name": "read",
                    "arguments": json.dumps({"path": "README.md"}),
                },
            }],
        },
        {
            "role": "tool",
            "name": "read",
            "tool_call_id": "c1",
            "content": "conteudo",
        },
    ]
    prompt = render_agent_prompt(messages, TOOLS)
    assert "conteudo" in prompt
    assert "README.md" in prompt
    assert "X" * 1000 not in prompt
