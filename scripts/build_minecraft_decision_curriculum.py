from __future__ import annotations

import json
from pathlib import Path

from conversa_llm.minecraft_decision import MINECRAFT_ACTION_TO_ID

TRAIN_CASES = [
    ("health=20 food=7 objective=survive inventory={bread:2} available=eat,wait", "eat"),
    ("health=20 food=20 objective=get_oak_log inventory={} available=gather,wait", "gather"),
    ("health=20 food=20 objective=get_cobblestone inventory={wooden_pickaxe:1} available=gather,wait", "gather"),
    ("health=20 food=20 objective=craft_stone_pickaxe inventory={cobblestone:3,stick:2} available=craft,gather,wait", "craft"),
    ("health=20 food=20 objective=craft_torch inventory={coal:1,stick:1} available=craft,wait", "craft"),
    ("health=20 food=20 objective=iron_ingot inventory={raw_iron:3,coal:1} available=smelt,wait", "smelt"),
    ("health=20 food=20 objective=glass inventory={sand:4,coal:1} available=smelt,wait", "smelt"),
    ("health=20 food=20 objective=go_home position={x:20,y:64,z:20} available=move,wait", "move"),
    ("health=20 food=20 objective=store_oak_log inventory={oak_log:32} available=deposit,wait", "deposit"),
    ("health=20 food=20 objective=get_sticks_from_storage inventory={} available=withdraw,gather,wait", "withdraw"),
    ("health=20 food=20 objective=build_shelter inventory={cobblestone:64} available=build,wait", "build"),
    ("health=12 food=18 objective=survive threat=zombie_near weapon=iron_sword available=fight,move,wait", "fight"),
    ("health=5 food=18 objective=survive threat=zombie_near escape_available=true available=fight,move,wait", "move"),
    ("health=20 food=20 objective=none threat=none available=wait", "wait"),
    ("health=20 food=20 objective=cancelled available=stop,wait", "stop"),
    ("health=20 food=6 objective=craft_pickaxe inventory={bread:1,cobblestone:3,stick:2} available=eat,craft,wait", "eat"),
    ("health=20 food=20 objective=craft_pickaxe inventory={cobblestone:1,stick:2} available=gather,craft,wait", "gather"),
    ("health=20 food=20 objective=smelt_iron inventory={raw_iron:0,coal:4} available=gather,smelt,wait", "gather"),
    ("health=20 food=20 objective=smelt_iron inventory={raw_iron:3,coal:0} available=gather,smelt,wait", "gather"),
    ("health=20 food=20 objective=deposit_cobblestone inventory={cobblestone:6} available=deposit,wait", "deposit"),
    ("health=20 food=7 objective=withdraw_food inventory={} available=withdraw,wait", "withdraw"),
    ("health=20 food=20 objective=resource_unreachable failures=1 alternative_route=true available=move,gather,wait", "move"),
    ("health=20 food=20 objective=worker_busy current_task=gather available=wait,stop", "wait"),
    ("health=20 food=20 objective=already_complete result=verified available=wait,stop", "wait"),
    ("health=20 food=20 objective=gather_stone failures=3 same_target=true available=gather,move,wait,stop", "move"),
    ("health=4 food=15 objective=survive threat=skeleton_near available=move,fight,wait", "move"),
    ("health=18 food=18 objective=defend threat=zombie_near weapon=stone_sword available=fight,move,wait", "fight"),
    ("health=20 food=20 objective=need_materials storage_has=iron_ingot available=withdraw,gather,wait", "withdraw"),
]

VALIDATION_CASES = [
    ("health=20 food=8 objective=continue_build inventory={bread:1} available=eat,build,wait", "eat"),
    ("health=20 food=20 objective=collect_birch_log inventory={} available=gather,wait", "gather"),
    ("health=20 food=20 objective=craft_chest inventory={oak_planks:8} available=craft,wait", "craft"),
    ("health=20 food=20 objective=make_charcoal inventory={oak_log:2,coal:0} available=smelt,gather,wait", "smelt"),
    ("health=20 food=20 objective=return_base position={x:-35,y:65,z:14} available=move,wait", "move"),
    ("health=20 food=20 objective=empty_inventory inventory={dirt:20} available=deposit,wait", "deposit"),
    ("health=20 food=20 objective=need_torches storage_has=torch available=withdraw,craft,wait", "withdraw"),
    ("health=15 food=19 objective=survive threat=husk_near weapon=iron_sword available=fight,move,wait", "fight"),
    ("health=20 food=20 objective=cancel_requested current_task=craft available=stop,wait", "stop"),
    ("health=20 food=20 objective=no_action_needed available=wait", "wait"),
]

BENCHMARK_CASES = [
    ("health=20 food=5 objective=mine_iron inventory={bread:2} available=eat,gather,wait", "eat"),
    ("health=20 food=20 objective=collect_spruce_log inventory={} available=gather,wait", "gather"),
    ("health=20 food=20 objective=craft_furnace inventory={cobblestone:8} available=craft,wait", "craft"),
    ("health=20 food=20 objective=smelt_gold inventory={raw_gold:2,coal:1} available=smelt,wait", "smelt"),
    ("health=20 food=20 objective=move_to_storage position={x:90,y:64,z:-30} available=move,wait", "move"),
    ("health=20 food=20 objective=store_raw_iron inventory={raw_iron:12} available=deposit,wait", "deposit"),
    ("health=20 food=20 objective=need_pickaxe storage_has=stone_pickaxe available=withdraw,craft,gather,wait", "withdraw"),
    ("health=14 food=18 objective=defend threat=zombie_near weapon=stone_axe available=fight,move,wait", "fight"),
    ("health=20 food=20 objective=cancelled_after_partial progress=3/6 available=stop,wait", "stop"),
    ("health=20 food=20 objective=task_verified_complete available=wait,stop", "wait"),
    ("health=6 food=16 objective=survive threat=creeper_near available=move,fight,wait", "move"),
    ("health=20 food=20 objective=resource_unreachable failures=4 alternative_route=false available=wait,stop", "stop"),
]


def rows(cases):
    return [{"state": state, "label": label} for state, label in cases]


def validate_splits():
    datasets = {
        "train": TRAIN_CASES,
        "validation": VALIDATION_CASES,
        "benchmark": BENCHMARK_CASES,
    }
    seen = {}
    for split, cases in datasets.items():
        for state, label in cases:
            if label not in MINECRAFT_ACTION_TO_ID:
                raise ValueError(f"ação inválida em {split}: {label}")
            if state in seen:
                raise ValueError(f"vazamento entre splits: {state!r} em {seen[state]} e {split}")
            seen[state] = split
    return datasets


def main():
    root = Path(__file__).resolve().parents[1]
    validate_splits()
    outputs = {
        "minecraft_decision_train.jsonl": rows(TRAIN_CASES) * 8,
        "minecraft_decision_validation.jsonl": rows(VALIDATION_CASES),
        "minecraft_decision_benchmark.jsonl": rows(BENCHMARK_CASES),
    }
    # Compatibilidade temporária com o nome antigo de treino.
    outputs["minecraft_decision_curriculum.jsonl"] = outputs["minecraft_decision_train.jsonl"]

    for name, values in outputs.items():
        path = root / "data" / name
        path.parent.mkdir(exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in values:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        "minecraft_train=%d minecraft_validation=%d minecraft_benchmark=%d"
        % (len(outputs["minecraft_decision_train.jsonl"]), len(VALIDATION_CASES), len(BENCHMARK_CASES))
    )


if __name__ == "__main__":
    main()
