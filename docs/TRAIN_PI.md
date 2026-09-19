# Treinar o checkpoint Conversa Pi

O alvo `cpu-8gb-pi` é o checkpoint mais capaz que o projeto tenta manter treinável em uma máquina com 8 GB de RAM: aproximadamente 6–7 milhões de parâmetros, contexto neural 512, 8 blocos Transformer e BPE próprio.

## 1. Gerar currículos e benchmarks

```powershell
python scripts/build_pi_curriculum.py
python scripts/build_pi_agent_curriculum.py
```

O primeiro currículo ensina chamadas diretas e Python curto. O segundo usa exatamente o prompt de produção do Pi e inclui sequências multi-turn:

```text
tarefa
→ read/grep/find/ls
→ resultado da tool
→ edit/write
→ resultado
→ bash/powershell
→ falha ou sucesso
→ corrigir novamente ou concluir
```

## 2. Treinar o byte-BPE

```powershell
python scripts/build_bpe_tokenizer.py `
  --data data/pi_curriculum.jsonl data/pi_agent_curriculum.jsonl `
  --out data/pi_bpe.json `
  --vocab-size 768
```

## 3. Treinar o modelo maior no perfil de 8 GB

```powershell
python -m conversa_llm.train `
  --profile cpu-8gb-pi `
  --tokenizer-file data/pi_bpe.json `
  --data data/pi_curriculum.jsonl data/pi_agent_curriculum.jsonl `
  --steps 3000 `
  --out checkpoints/conversa-pi.pt
```

Esse treino é bem mais pesado que o modelo anterior de centenas de milhares de parâmetros. Em CPU ele pode ser demorado, mas a configuração foi escolhida para caber em 8 GB. Para uma primeira validação, use `--steps 200`.

Se houver pressão de memória, reduza a arquitetura explicitamente:

```powershell
python -m conversa_llm.train `
  --tokenizer-file data/pi_bpe.json `
  --data data/pi_curriculum.jsonl data/pi_agent_curriculum.jsonl `
  --d-model 192 `
  --layers 6 `
  --heads 6 `
  --context 384 `
  --batch-size 1 `
  --grad-accum 8 `
  --steps 2000 `
  --out checkpoints/conversa-pi.pt
```

## 4. Avaliar chamadas isoladas

```powershell
python -m conversa_llm.eval_tools `
  --model checkpoints/conversa-pi.pt `
  --tokenizer-file data/pi_bpe.json `
  --verbose
```

## 5. Avaliar estados multi-turn reais

```powershell
python -m conversa_llm.eval_agent `
  --model checkpoints/conversa-pi.pt `
  --tokenizer-file data/pi_bpe.json `
  --verbose
```

Esse benchmark verifica decisões após resultados de ferramentas, inclusive o passo "editar → testar" e "teste passou → concluir".

## 6. Avaliar roteador híbrido

```powershell
python scripts/eval_hybrid_tools.py
```

O roteador é uma camada determinística para pedidos explícitos. A capacidade neural deve ser analisada separadamente com `eval_tools` e `eval_agent`.

## 7. Avaliar código executável

```powershell
python -m conversa_llm.eval_code `
  --model checkpoints/conversa-pi.pt `
  --tokenizer-file data/pi_bpe.json `
  --verbose
```

## Barra de capacidade

Não considere o modelo "capaz" apenas porque gera JSON válido. O marco de agente exige, no mínimo:

- escolher a ferramenta correta;
- copiar argumentos corretamente;
- usar resultados de ferramentas no turno seguinte;
- corrigir uma primeira tentativa quando testes falham;
- terminar quando os testes passam;
- gerar Python que compile e passe testes inéditos;
- evitar chamadas desnecessárias quando já pode responder.

Os percentuais dos benchmarks são locais ao projeto e não equivalem a uma porcentagem da capacidade de modelos comerciais.
