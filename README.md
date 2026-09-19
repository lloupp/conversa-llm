# Conversa LLM

Um LLM conversacional pequeno construído **do zero**, sem pesos pré-treinados e sem tokenizer externo.

O objetivo inicial não é competir com modelos comerciais. É construir a cadeia completa e entendê-la:

`dados → tokenizer → Transformer causal → treinamento → checkpoint → geração → chat`

## Estado atual — marco 0.1

- tokenizer byte-level próprio (UTF-8);
- tokens especiais para usuário e assistente;
- Transformer causal implementado em PyTorch;
- atenção causal, embeddings, blocos Transformer e MLP;
- treinamento supervisionado em JSONL;
- perda aplicada às respostas do assistente;
- geração autoregressiva com temperature e top-k;
- CLI de conversa;
- testes unitários;
- corpus mínimo apenas para validar o pipeline.

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\\Scripts\\activate  # Windows
pip install -e .[dev]
```

## Testes

```bash
pytest -q
```

## Treinar um modelo pequeno

```bash
python -m conversa_llm.train \
  --data data/sample_conversations.jsonl \
  --steps 300 \
  --out checkpoints/model.pt
```

### PC com 8 GB de RAM

O projeto tem um perfil conservador para máquinas sem muita memória:

```bash
python -m conversa_llm.train \
  --profile cpu-8gb \
  --data data/sample_conversations.jsonl data/python_conversations.jsonl \
  --steps 300 \
  --out checkpoints/cpu-8gb.pt
```

Esse perfil usa contexto 128, `d_model=96`, 3 camadas, batch 2 e acumulação de gradiente 8. Assim o batch efetivo é 16 sem manter 16 exemplos simultaneamente na RAM.

Para um teste ainda menor:

```bash
python -m conversa_llm.train \
  --steps 100 \
  --d-model 64 \
  --layers 2 \
  --heads 4 \
  --context 128 \
  --batch-size 1 \
  --grad-accum 4 \
  --out checkpoints/tiny.pt
```

## Conversar

```bash
python -m conversa_llm.chat --model checkpoints/model.pt
```

## Formato dos dados

Cada linha do JSONL contém uma conversa:

```json
{"messages":[
  {"role":"user","content":"Olá"},
  {"role":"assistant","content":"Olá! Como posso ajudar?"}
]}
```

### Ensinar Python sem apagar a conversa geral

O treinador aceita vários arquivos JSONL. Para misturar conversa geral com o currículo inicial de Python:

```bash
python -m conversa_llm.train \
  --data data/sample_conversations.jsonl data/python_conversations.jsonl \
  --steps 1000 \
  --out checkpoints/python-model.pt
```

O arquivo `data/python_conversations.jsonl` é apenas a semente do currículo. Um modelo útil em programação exigirá um corpus muito maior, diversificado, deduplicado e com licenças compatíveis.

Veja também `docs/HARDWARE.md`.

## O que significa “do zero” neste projeto

Nenhum peso de outro LLM é carregado. O tokenizer não depende de GPT, Llama, BERT ou outro modelo. Os parâmetros começam aleatórios e aprendem somente com os dados fornecidos ao treinamento.

PyTorch é usado como biblioteca numérica e de autodiferenciação. Em etapas futuras, partes internas poderão ser implementadas em nível ainda mais baixo para fins didáticos.

## Próximos marcos

1. tokenizer BPE próprio e treinável;
2. RoPE e RMSNorm;
3. dataset maior em português e pipeline de limpeza;
4. pretraining não supervisionado antes do ajuste conversacional;
5. avaliação automática de perplexidade e qualidade;
6. checkpoints retomáveis, mixed precision e GPU;
7. interface web e API de inferência;
8. memória de conversa e contexto longo;
9. treinamento distribuído quando a escala justificar.

## Marco 0.2 — Python acima da meta de 20%

A meta foi redefinida de forma verificável: o modelo precisa responder por **geração livre greedy**, sem alternativas, a 50 perguntas de Python cuja formulação não aparece no currículo de treino.

O primeiro modelo byte-level decorava exemplos mas generalizava mal. A solução foi adicionar um `WordTokenizer` treinado **do zero no próprio corpus**. Nenhum tokenizer ou peso externo é usado.

Resultado atual:

- tokenizer: word-level próprio, 557 tokens;
- modelo: 139.328 parâmetros;
- arquitetura: Transformer causal, `d_model=64`, 2 camadas, 4 heads;
- contexto: 64 tokens;
- treino: 1.000 passos;
- loss: 6,3432 → 0,0298;
- benchmark de geração Python: **29/50 = 58,0%**;
- meta pedida: 20%.

O número de 58% significa 58% deste benchmark interno de 50 habilidades básicas, **não 58% de todo o conhecimento de Python de um modelo grande**. Ainda há bastante trabalho em explicações longas, código novo, depuração e generalização para formulações muito diferentes.

### Reproduzir o currículo e tokenizer

```bash
python scripts/build_python_curriculum.py
PYTHONPATH=src python scripts/build_word_tokenizer.py \
  --data data/python_recall_curriculum.jsonl \
  --out data/python_word_vocab.json
```

No Windows PowerShell, depois de `pip install -e .[dev]`, você pode executar os mesmos módulos sem `PYTHONPATH=src`.

### Reproduzir o treino que superou 20%

```bash
python -m conversa_llm.train \
  --tokenizer-file data/python_word_vocab.json \
  --data data/python_recall_curriculum.jsonl \
  --steps 1000 \
  --d-model 64 \
  --layers 2 \
  --heads 4 \
  --context 64 \
  --batch-size 8 \
  --grad-accum 1 \
  --lr 0.001 \
  --out checkpoints/python-word-1000.pt
```

### Avaliar por geração livre

```bash
python -m conversa_llm.eval_generation \
  --tokenizer-file data/python_word_vocab.json \
  --model checkpoints/python-word-1000.pt \
  --benchmark data/python_generation_benchmark.jsonl \
  --verbose
```

### Conversar com esse checkpoint

```bash
python -m conversa_llm.chat \
  --model checkpoints/python-word-1000.pt \
  --tokenizer-file data/python_word_vocab.json
```

## Próximo marco

O próximo salto é trocar o word-level por **BPE próprio**, aumentar o corpus de Python com exemplos licenciados e separar três avaliações: conhecimento, geração de código executável e depuração. A meta seguinte deve ser 70% no benchmark atual sem sacrificar conversação geral.
