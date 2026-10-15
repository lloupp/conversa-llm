"""Coleta exemplos supervisionados da API NVIDIA para o Conversa LLM.

Somente utilize saídas para treino após verificar a licença do modelo
selecionado e as condições de uso da API. Não executa ferramentas/código.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ACTIONS = (
    "answer", "web_search", "read", "write", "edit", "bash", "powershell",
    "grep", "find", "ls", "stop",
)
API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{number}: esperado objeto JSON")
                records.append(record)
    return records


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_seed(seed: dict) -> dict:
    kind = seed.get("kind")
    if kind not in {"generation", "decision"}:
        raise ValueError("kind precisa ser generation ou decision")
    field = "prompt" if kind == "generation" else "state"
    value = seed.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{kind} precisa de {field} não vazio")
    if kind == "decision" and seed.get("expected_label") not in ACTIONS:
        raise ValueError("decisões exigem expected_label pertencente a ACTIONS")
    identity = json.dumps([kind, value], ensure_ascii=False, sort_keys=True)
    return {**seed, "id": hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]}


def prepare_seeds(seeds: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique: list[dict] = []
    for seed in seeds:
        if seed.get("split", "train") != "train":
            continue
        clean = validate_seed(seed)
        if clean["id"] not in seen:
            seen.add(clean["id"])
            unique.append(clean)
    return unique


def teacher_messages(seed: dict) -> list[dict]:
    if seed["kind"] == "generation":
        return [
            {"role": "system", "content": (
                "Responda em português do Brasil de forma correta, concisa e autossuficiente. "
                "Se solicitarem código, devolva apenas código executável, sem blocos Markdown. "
                "Não invente resultados de ferramentas ou testes."
            )},
            {"role": "user", "content": seed["prompt"]},
        ]
    return [
        {"role": "system", "content": (
            "Classifique o PRÓXIMO passo do agente usando EXATAMENTE uma palavra "
            "desta lista, sem explicar: " + ", ".join(ACTIONS) + ". "
            "Nunca execute ferramentas. 'stop' somente se a tarefa terminou."
        )},
        {"role": "user", "content": seed["state"]},
    ]


def response_content(payload: dict) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("resposta da API sem choices")
    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    if not isinstance(content, str):
        raise ValueError("resposta da API sem conteúdo textual")
    return content.strip()


def nvidia_chat(api_key: str, model: str, messages: list[dict],
                max_tokens: int, timeout: float, retries: int) -> dict:
    payload = json.dumps({
        "model": model, "messages": messages, "temperature": 0.2,
        "max_tokens": max_tokens, "stream": False,
    }).encode("utf-8")
    request = urllib.request.Request(
        API_URL, data=payload, method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as result:
                return json.loads(result.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            retryable = exc.code == 429 or 500 <= exc.code < 600
            if not retryable or attempt == retries:
                raise RuntimeError(f"NVIDIA HTTP {exc.code}") from None
            retry_after = exc.headers.get("Retry-After", "")
            delay = min(30.0, float(retry_after)) if retry_after.replace(".", "", 1).isdigit() else min(2 ** attempt, 30)
            time.sleep(max(0.5, delay))
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries:
                raise RuntimeError("Falha de conexão com a API NVIDIA") from None
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError("Falha inesperada da API")


def normalized_answer(seed: dict, answer: str, max_chars: int) -> tuple[dict | None, str]:
    if not answer or len(answer) > max_chars:
        return None, "resposta vazia ou acima do limite"
    if seed["kind"] == "decision":
        label = answer.strip().strip("'\" ").lower()
        if label not in ACTIONS:
            return None, "label desconhecido ou resposta fora do formato"
        if label != seed["expected_label"]:
            return None, f"divergência do gabarito: {label}"
        return {"state": seed["state"], "label": label}, "aceito"
    return {"prompt": seed["prompt"], "completion": answer}, "aceito"


def run_collection(seeds: list[dict], out_dir: Path, teacher: str, limit: int,
                   max_tokens: int, max_chars: int, api_key: str,
                   client=nvidia_chat, delay: float = 0.5, timeout: float = 90,
                   retries: int = 2) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.jsonl"
    prior = read_jsonl(manifest) if manifest.exists() else []
    if any(item.get("teacher") != teacher for item in prior):
        raise ValueError("diretório contém dados de outro professor; use outro --out-dir")
    done = {item["id"] for item in prior if "id" in item}
    totals = {"processed": 0, "accepted": 0, "rejected": 0, "skipped": 0,
              "prompt_tokens": 0, "completion_tokens": 0}
    for seed in seeds:
        if seed["id"] in done:
            totals["skipped"] += 1
            continue
        if totals["processed"] >= limit:
            break
        reply = client(api_key, teacher, teacher_messages(seed), max_tokens, timeout, retries)
        answer = response_content(reply)
        accepted, reason = normalized_answer(seed, answer, max_chars)
        usage = reply.get("usage") or {}
        for usage_field in ("prompt_tokens", "completion_tokens"):
            value = usage.get(usage_field, 0)
            totals[usage_field] += value if isinstance(value, int) and value >= 0 else 0
        if accepted:
            target = ("generation_train.jsonl" if seed["kind"] == "generation"
                      else "decision_train.jsonl")
            append_jsonl(out_dir / target, accepted)
            totals["accepted"] += 1
        else:
            totals["rejected"] += 1
        append_jsonl(manifest, {
            "id": seed["id"], "kind": seed["kind"], "teacher": teacher,
            "status": "accepted" if accepted else "rejected", "reason": reason,
            "usage": usage,
        })
        done.add(seed["id"])
        totals["processed"] += 1
        print(f"[{totals['processed']}/{limit}] {seed['kind']} {reason}", flush=True)
        if delay > 0 and totals["processed"] < limit:
            time.sleep(delay)
    return totals


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="data/nvidia_distill_seeds.jsonl")
    parser.add_argument("--out-dir", default="data/distilled_nvidia")
    parser.add_argument("--teacher", help="ID exato do modelo autorizado na NVIDIA")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=180)
    parser.add_argument("--max-chars", type=int, default=800)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--license-confirmed", action="store_true", help=(
        "Declara que você verificou as condições da API e da licença específica do "
        "modelo, incluindo permissão de usar saídas para treinar seu próprio modelo"
    ))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.limit < 1 or args.max_tokens < 1 or args.max_chars < 1 or args.retries < 0 or args.delay < 0:
        raise ValueError("limit/max-tokens/max-chars >= 1; retries/delay >= 0")
    seeds = prepare_seeds(read_jsonl(Path(args.seeds)))
    kinds = {kind: sum(x["kind"] == kind for x in seeds) for kind in ("generation", "decision")}
    print(f"Seeds únicos de treino: {len(seeds)} {kinds}; limite de chamadas: {args.limit}")
    if args.dry_run:
        print("DRY-RUN: nenhuma chamada à NVIDIA, nenhum dado de treino produzido")
        return 0
    if not args.license_confirmed:
        raise ValueError("Confira as licenças e passe --license-confirmed se autorizado")
    if not args.teacher:
        raise ValueError("Selecione um modelo explícito com --teacher")
    api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Defina NVIDIA_API_KEY nas variáveis de ambiente/Secrets do Colab")
    totals = run_collection(seeds, Path(args.out_dir), args.teacher, args.limit,
                            args.max_tokens, args.max_chars, api_key,
                            delay=args.delay, timeout=args.timeout, retries=args.retries)
    print(json.dumps(totals, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        print(f"Erro: {error}", file=sys.stderr)
        sys.exit(1)
