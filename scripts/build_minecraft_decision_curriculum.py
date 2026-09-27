from __future__ import annotations

import json
from pathlib import Path

CASES = [
("health=20 food=9 objective=survive inventory={bread:2}", "eat"),
("health=20 food=20 objective=get_oak_log inventory={}", "gather"),
("health=20 food=20 objective=get_cobblestone inventory={wooden_pickaxe:1}", "gather"),
("health=20 food=20 objective=craft_stone_pickaxe inventory={cobblestone:3,stick:2}", "craft"),
("health=20 food=20 objective=craft_torch inventory={coal:1,stick:1}", "craft"),
("health=20 food=20 objective=iron_ingot inventory={raw_iron:3,coal:1}", "smelt"),
("health=20 food=20 objective=glass inventory={sand:4,coal:1}", "smelt"),
("health=20 food=20 objective=go_home position={x:20,y:64,z:20}", "move"),
("health=20 food=20 objective=store_oak_log inventory={oak_log:32}", "deposit"),
("health=20 food=20 objective=get_sticks_from_storage inventory={}", "withdraw"),
("health=20 food=20 objective=build_shelter inventory={cobblestone:64}", "build"),
("health=12 food=18 objective=survive hostile=zombie_near weapon=iron_sword", "fight"),
("health=5 food=8 objective=survive hostile=zombie_near escape_available=true", "move"),
("health=20 food=20 objective=none hostile=none", "wait"),
("health=20 food=20 objective=cancelled", "stop"),
("health=20 food=6 objective=craft_pickaxe inventory={bread:1,cobblestone:3,stick:2}", "eat"),
("health=20 food=20 objective=craft_pickaxe inventory={cobblestone:1,stick:2}", "gather"),
("health=20 food=20 objective=smelt_iron inventory={raw_iron:0,coal:4}", "gather"),
("health=20 food=20 objective=deposit_cobblestone inventory={cobblestone:6}", "deposit"),
("health=20 food=20 objective=withdraw_food food=7 inventory={}", "withdraw"),
]


def main():
    root = Path(__file__).resolve().parents[1]
    rows = [{"state": state, "label": label} for state, label in CASES]
    train = rows * 8
    benchmark = rows
    for name, values in [("minecraft_decision_curriculum.jsonl", train), ("minecraft_decision_benchmark.jsonl", benchmark)]:
        path = root / "data" / name
        path.parent.mkdir(exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in values:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"minecraft_train={len(train)} minecraft_benchmark={len(benchmark)}")


if __name__ == "__main__":
    main()
