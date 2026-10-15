import json
from pathlib import Path

import pytest

from scripts.distill_nvidia import (
    normalized_answer, prepare_seeds, response_content, run_collection,
    teacher_messages, validate_seed,
)


def test_dedup_does_not_include_holdout():
    seeds = [
        {"kind": "generation", "prompt": "Olá"},
        {"kind": "generation", "prompt": "Olá"},
        {"kind": "generation", "prompt": "Teste", "split": "eval"},
    ]
    items = prepare_seeds(seeds)
    assert len(items) == 1
    assert len(items[0]["id"]) == 20


def test_malformed_decision_seed_rejected():
    with pytest.raises(ValueError):
        validate_seed({"kind": "decision", "state": "USER: test", "expected_label": "execute"})


def test_teacher_decision_labels_checked():
    seed = {"kind": "decision", "state": "USER: Leia readme", "expected_label": "read"}
    assert normalized_answer(seed, "read", 800)[0] == {"state": seed["state"], "label": "read"}
    assert normalized_answer(seed, "bash", 800)[0] is None
    assert normalized_answer(seed, "read\nporque quero", 800)[0] is None
    assert "Nunca execute" in teacher_messages(seed)[0]["content"]


def test_generation_filters_empty_or_long():
    seed = {"kind": "generation", "prompt": "Explique"}
    assert normalized_answer(seed, "", 3)[0] is None
    assert normalized_answer(seed, "quatro", 3)[0] is None
    assert normalized_answer(seed, "ok", 3)[0] == {"prompt": "Explique", "completion": "ok"}


def test_api_text_response():
    assert response_content({"choices": [{"message": {"content": " ok "}}]}) == "ok"
    assert response_content({"choices": [{"message": {"content": [{"text": "olá"}]}}]}) == "olá"
    with pytest.raises(ValueError):
        response_content({"choices": []})


def test_offline_collection_and_resume(tmp_path: Path):
    seeds = prepare_seeds([
        {"kind": "generation", "prompt": "Escreva teste"},
        {"kind": "decision", "state": "USER: Liste src", "expected_label": "ls"},
        {"kind": "decision", "state": "USER: Execute pytest", "expected_label": "bash"},
    ])
    answers = iter(["def teste(): pass", "ls", "write"])
    calls = []

    def fake_client(key, model, messages, max_tokens, timeout, retries):
        calls.append((key, model, messages))
        return {"choices": [{"message": {"content": next(answers)}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2}}

    first = run_collection(seeds, tmp_path, "teacher-A", 2, 180, 800, "fake-key",
                           client=fake_client, delay=0)
    assert first["accepted"] == 2
    second = run_collection(seeds, tmp_path, "teacher-A", 2, 180, 800, "fake-key",
                            client=fake_client, delay=0)
    assert second["accepted"] == 0 and second["rejected"] == 1
    assert second["skipped"] == 2
    assert len(calls) == 3
    assert len((tmp_path / "manifest.jsonl").read_text().splitlines()) == 3
    assert len((tmp_path / "generation_train.jsonl").read_text().splitlines()) == 1
    assert len((tmp_path / "decision_train.jsonl").read_text().splitlines()) == 1
    assert json.loads((tmp_path / "decision_train.jsonl").read_text())["label"] == "ls"
    with pytest.raises(ValueError):
        run_collection(seeds, tmp_path, "teacher-B", 1, 180, 800, "fake-key",
                       client=fake_client, delay=0)
