# Conversa-LLM decision engine (experimental)

The Minecraft profile is intentionally separate from the coding-agent action head.

## Prepare and train

```bash
python scripts/build_minecraft_decision_curriculum.py
python scripts/build_bpe_tokenizer.py --data data/minecraft_decision_curriculum.jsonl --out /tmp/minecraft-bpe.json --vocab-size 300 --min-pair-freq 2
python -m conversa_llm.train_minecraft_decision --data data/minecraft_decision_curriculum.jsonl --tokenizer-file /tmp/minecraft-bpe.json --out checkpoints/conversa-minecraft.pt
```

The bundled curriculum is only a bootstrap dataset. Do not treat its training accuracy as runtime validation.

## Start local decision service

```bash
python -m conversa_llm.minecraft_server --model checkpoints/conversa-minecraft.pt --tokenizer-file /tmp/minecraft-bpe.json --host 127.0.0.1 --port 8766 --threshold 0.70
```

Endpoint: `POST /v1/minecraft/decision`, body `{"state": {...}}`.

The service returns an action, confidence, probabilities and `trusted`. Keep it bound to localhost. The Minecraft bot must independently validate the action and fall back when the service is unavailable, slow, invalid or below threshold.
