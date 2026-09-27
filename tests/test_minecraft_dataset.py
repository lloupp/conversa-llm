from pathlib import Path
import runpy

from conversa_llm.minecraft_decision import MINECRAFT_ACTION_TO_ID

_module = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "build_minecraft_decision_curriculum.py"))
TRAIN_CASES = _module["TRAIN_CASES"]
VALIDATION_CASES = _module["VALIDATION_CASES"]
BENCHMARK_CASES = _module["BENCHMARK_CASES"]
validate_splits = _module["validate_splits"]


def test_minecraft_dataset_splits_are_disjoint():
    validate_splits()
    train = {state for state, _ in TRAIN_CASES}
    validation = {state for state, _ in VALIDATION_CASES}
    benchmark = {state for state, _ in BENCHMARK_CASES}
    assert train.isdisjoint(validation)
    assert train.isdisjoint(benchmark)
    assert validation.isdisjoint(benchmark)


def test_minecraft_dataset_labels_are_valid_actions():
    for cases in (TRAIN_CASES, VALIDATION_CASES, BENCHMARK_CASES):
        assert all(label in MINECRAFT_ACTION_TO_ID for _, label in cases)


def test_benchmark_contains_safety_and_recovery_cases():
    states = " ".join(state for state, _ in BENCHMARK_CASES)
    assert "cancelled_after_partial" in states
    assert "resource_unreachable" in states
    assert "creeper_near" in states
