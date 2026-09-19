from conversa_llm.eval_python import _candidate_example
from conversa_llm.tokenizer import ByteTokenizer


def test_candidate_example_supervises_only_answer():
    tok = ByteTokenizer()
    inputs, labels = _candidate_example(tok, "Pergunta?", "Use len(...).", 128)
    supervised = [label for label in labels if label != -100]
    assert supervised == tok.encode_text("Use len(...).") + [tok.SEP]
    assert len(inputs) == len(labels)


def test_candidate_example_respects_context():
    tok = ByteTokenizer()
    inputs, labels = _candidate_example(tok, "x" * 400, "Use %.", 64)
    assert len(inputs) <= 64
    assert len(labels) == len(inputs)
