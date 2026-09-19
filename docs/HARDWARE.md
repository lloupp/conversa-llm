# Hardware para treinar o Conversa LLM

O requisito depende muito mais do tamanho do modelo e do contexto do que da interface de chat.

## Perfil `cpu-8gb` — seu caso

- 8 GB de RAM;
- CPU moderna;
- GPU não é obrigatória;
- SSD com pelo menos 10 GB livres.

O perfil usa um modelo pequeno: `d_model=96`, 3 camadas, 4 heads, contexto 128, batch 2 e gradient accumulation 8. Ele serve para desenvolver a arquitetura, validar datasets, ensinar conceitos básicos de Python e produzir checkpoints experimentais.

Execute:

```bash
python -m conversa_llm.train \
  --profile cpu-8gb \
  --data data/sample_conversations.jsonl data/python_conversations.jsonl \
  --steps 300 \
  --out checkpoints/cpu-8gb.pt
```

Feche programas pesados durante o treino. Se houver pressão de memória, reduza `--batch-size` para 1; `--grad-accum` pode ser aumentado para preservar um batch efetivo maior.

## Perfil `cpu-16gb`

- 16 GB de RAM;
- CPU moderna de 6+ núcleos;
- GPU opcional;
- SSD com 20+ GB livres.

Permite contexto e modelo um pouco maiores, ainda visando desenvolvimento e experimentação.

## Perfil `gpu-12gb`

- 32 GB de RAM recomendado;
- NVIDIA com 12 GB ou mais de VRAM;
- SSD NVMe.

É um salto importante para pretraining mais longo e modelos pequenos com dezenas de milhões de parâmetros.

## Escala maior

Modelos realmente grandes deixam de ser um projeto de uma máquina doméstica. A partir daí entram GPUs alugadas, múltiplas GPUs e treinamento distribuído.

## Regra prática

Comece pequeno, meça o uso real e aumente uma dimensão por vez. Em CPU com 8 GB, contexto, batch e número de camadas precisam permanecer conservadores. Checkpoints e datasets grandes também devem ficar fora do histórico normal do Git.
