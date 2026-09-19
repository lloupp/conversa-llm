# Treinar o checkpoint Conversa Pi

## 1. Gerar currículo e benchmarks

```powershell
python scripts/build_pi_curriculum.py
```

## 2. Treinar o byte-BPE

```powershell
python scripts/build_bpe_tokenizer.py `
  --data data/pi_curriculum.jsonl `
  --out data/pi_bpe.json `
  --vocab-size 512
```

## 3. Treinar no perfil de 8 GB

```powershell
python -m conversa_llm.train `
  --profile cpu-8gb-pi `
  --tokenizer-file data/pi_bpe.json `
  --data data/pi_curriculum.jsonl `
  --steps 1500 `
  --out checkpoints/conversa-pi.pt
```

Se faltar memória:

```powershell
python -m conversa_llm.train `
  --profile cpu-8gb-pi `
  --tokenizer-file data/pi_bpe.json `
  --data data/pi_curriculum.jsonl `
  --batch-size 1 `
  --grad-accum 8 `
  --steps 1500 `
  --out checkpoints/conversa-pi.pt
```

## 4. Avaliar tools neurais

```powershell
python -m conversa_llm.eval_tools `
  --model checkpoints/conversa-pi.pt `
  --tokenizer-file data/pi_bpe.json `
  --verbose
```

## 5. Avaliar roteador híbrido

```powershell
python scripts/eval_hybrid_tools.py
```

## 6. Avaliar código executável

```powershell
python -m conversa_llm.eval_code `
  --model checkpoints/conversa-pi.pt `
  --tokenizer-file data/pi_bpe.json `
  --verbose
```

Os números de benchmark devem ser interpretados somente dentro desses conjuntos pequenos. Eles não representam uma porcentagem de capacidade geral de um LLM grande.
