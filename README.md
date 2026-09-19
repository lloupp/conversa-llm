# Conversa LLM

Um modelo conversacional e agente de código pequeno, construído do zero, sem pesos pré-treinados.

O projeto evoluiu para uma arquitetura híbrida inspirada no conceito **System One**: uma camada de decisão escolhe ações tipadas com probabilidades/confiança, enquanto um Transformer generativo escreve apenas quando necessário.

## Componentes

- byte tokenizer e byte-BPE próprios;
- Transformer causal treinado do zero;
- modelo System One de decisão separado;
- decisões: answer, web_search, read, write, edit, bash, powershell, grep, find, ls e stop;
- probabilidades, confidence gate, Brier score e ECE;
- geração de Python;
- tool calling OpenAI-compatible para Pi Agent;
- fluxo multi-turn com resultados de ferramentas;
- fallback web;
- perfis para máquinas com 8 GB de RAM.

## Documentação

- `docs/PI_AGENT.md`: usar dentro do Pi Agent;
- `docs/TRAIN_PI.md`: treinar o checkpoint generativo;
- `docs/SYSTEM_ONE.md`: arquitetura inspirada no Jev e treino da camada de decisão;
- `docs/HARDWARE.md`: perfis de hardware.

## Princípio

```text
estado
  ↓
System One decision model
  ↓
ação + probabilidades + confiança
  ↓
código/harness decide o caminho
  ↓
Transformer generativo somente quando necessário
```

A arquitetura interna do Jev não é pública. Este projeto reproduz o princípio de separação entre decisões estruturadas e geração, não o modelo proprietário da TypeSafe.

## Instalação

```bash
python -m venv .venv
pip install -e ".[dev,web]"
pytest -q
```

Consulte a documentação acima para treino e integração no Pi.
