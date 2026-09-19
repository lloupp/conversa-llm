from __future__ import annotations

import json
from pathlib import Path

from conversa_llm.pi_tools import PI_CORE_TOOLS
from conversa_llm.tool_router import route_explicit_tool


def main() -> None:
    path = Path("data/pi_tool_benchmark.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    exact = 0
    for row in rows:
        call = route_explicit_tool(row["request"], PI_CORE_TOOLS)
        ok = bool(call and call.name == row["name"] and call.arguments == row["arguments"])
        exact += int(ok)
        print(f"[{'OK' if ok else '--'}] {row['request']}")
    print(f"hybrid_exact={exact}/{len(rows)}={100*exact/max(1,len(rows)):.1f}%")


if __name__ == "__main__":
    main()
