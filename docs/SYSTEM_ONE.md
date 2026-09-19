# Arquitetura System One do Conversa Pi

O projeto é inspirado no conceito público do Jev/TypeSafe, não na arquitetura interna do Jev.

A TypeSafe descreve Jev como um modelo que recebe estado não estruturado e produz decisões tipadas com probabilidades/confiança, em vez de gerar strings autoregressivamente. A arquitetura, pesos, quantidade de parâmetros e dados de treino do Jev não são públicos.

## Arquitetura do Conversa Pi

O sistema passa a ter dois modelos:

```text
estado do agente
      │
      ▼
DecisionModel (System One)
      │
      ├── answer
      ├── web_search
      ├── read
      ├── write
      ├── edit
      ├── bash
      ├── powershell
      ├── grep
      ├── find
      ├── ls
      └── stop
      │
      ▼
probabilidades + confiança
      │
      ├── confiança alta + tool → executar/produzir argumentos
      ├── web_search → busca externa
      └── answer/stop/baixa confiança → Transformer generativo
```

O modelo de decisão faz somente uma classificação por requisição. Ele não escreve código ou prosa.

O Transformer generativo continua responsável por:

- texto livre;
- código Python;
- argumentos mais complexos de ferramentas;
- conclusão após uma sequência de ferramentas.

## Por que separar

Um gerador autoregressivo precisa escrever tokens mesmo quando a resposta real é apenas "qual ferramenta usar?". A camada System One reduz esse problema para uma decisão fechada.

Isso também permite medir separadamente:

- acurácia da decisão;
- NLL;
- Brier score;
- ECE (Expected Calibration Error);
- confiança mínima para automação.

## Calibração

Após treinar:

```powershell
python -m conversa_llm.calibrate_decision `
  --model checkpoints/conversa-decision.pt `
  --data data/decision_benchmark.jsonl `
  --tokenizer-file data/pi_bpe.json `
  --out checkpoints/conversa-decision-calibrated.pt
```

Depois avalie:

```powershell
python -m conversa_llm.eval_decision `
  --model checkpoints/conversa-decision-calibrated.pt `
  --tokenizer-file data/pi_bpe.json `
  --data data/decision_benchmark.jsonl
```

## Servidor Pi

Suba os dois checkpoints:

```powershell
python -m conversa_llm.openai_server `
  --model checkpoints/conversa-pi.pt `
  --tokenizer-file data/pi_bpe.json `
  --decision-model checkpoints/conversa-decision-calibrated.pt `
  --decision-tokenizer-file data/pi_bpe.json `
  --decision-threshold 0.70 `
  --web-fallback
```

O threshold é uma política do harness. Quanto maior, mais decisões incertas caem para o caminho generativo/híbrido.

## O que ainda não reproduz Jev

- sampler paralelo proprietário;
- RLCD;
- arquitetura interna;
- garantia matemática de schema do modelo em si;
- calibração de fronteira;
- inteligência comparável à versão comercial.

O objetivo é adotar a separação arquitetural entre **decisão fechada** e **geração aberta**, com métricas explícitas de incerteza.
