from pathlib import Path

from conversa_llm.data import ConversationDataset


def test_dataset_accepts_multiple_files(tmp_path: Path):
    payload = '{"messages":[{"role":"user","content":"oi"},{"role":"assistant","content":"olá"}]}\n'
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    first.write_text(payload, encoding="utf-8")
    second.write_text(payload, encoding="utf-8")

    dataset = ConversationDataset([first, second], context_length=64)

    assert len(dataset) == 2


def test_dataset_can_supervise_full_sequence(tmp_path: Path):
    payload = '{"messages":[{"role":"user","content":"abc"},{"role":"assistant","content":"xyz"}]}\n'
    path = tmp_path / "full.jsonl"
    path.write_text(payload, encoding="utf-8")
    normal = ConversationDataset(path, context_length=64)
    full = ConversationDataset(path, context_length=64, supervise_all=True)
    assert sum(x != -100 for x in full[0][1]) > sum(x != -100 for x in normal[0][1])
