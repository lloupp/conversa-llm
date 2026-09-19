from conversa_llm.openai_server import latest_user_prompt, message_text, split_stream_text


def test_message_text_string():
    assert message_text("oi") == "oi"


def test_message_text_openai_parts():
    content = [
        {"type": "text", "text": "primeira"},
        {"type": "input_text", "text": "segunda"},
        {"type": "image_url", "image_url": {"url": "x"}},
    ]
    assert message_text(content) == "primeira\nsegunda"


def test_latest_user_prompt_ignores_system_and_assistant():
    messages = [
        {"role": "system", "content": "sistema"},
        {"role": "user", "content": "pergunta antiga"},
        {"role": "assistant", "content": "resposta"},
        {"role": "user", "content": "pergunta nova"},
    ]
    assert latest_user_prompt(messages) == "pergunta nova"


def test_stream_text_reassembles():
    text = "abcdefghijklmnopqrstuvwxyz"
    assert "".join(split_stream_text(text, 5)) == text
