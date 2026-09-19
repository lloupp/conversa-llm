# Usar o Conversa LLM no Pi Agent

O Pi aceita providers locais compatíveis com OpenAI Chat Completions. O Conversa LLM expõe:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- respostas normais e streaming SSE

## 1. Instalar

No repositório:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

Para habilitar fallback web:

```powershell
pip install -e ".[web]"
```

## 2. Colocar o checkpoint

Exemplo:

```text
checkpoints/python-58pct.pt
data/python_word_vocab.json
```

## 3. Subir a API local

```powershell
python -m conversa_llm.openai_server `
  --model checkpoints/python-58pct.pt `
  --tokenizer-file data/python_word_vocab.json
```

Com fallback web:

```powershell
python -m conversa_llm.openai_server `
  --model checkpoints/python-58pct.pt `
  --tokenizer-file data/python_word_vocab.json `
  --web-fallback
```

Teste:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

## 4. Configurar o Pi

O arquivo de modelos do Pi fica em:

```text
~/.pi/agent/models.json
```

No Windows:

```text
$HOME\.pi\agent\models.json
```

Use `examples/pi/models.json` como provider ou mescle a entrada `conversa-local` no seu arquivo existente.

Depois abra o Pi e use `/model` para selecionar:

```text
Conversa LLM Python 58 (local)
```

## Limite atual importante

O checkpoint Python 58 tem contexto neural efetivo de apenas 64 tokens. O adapter anuncia uma janela maior para que o Pi consiga enviar seu envelope de sistema, mas por enquanto **somente a mensagem de usuário mais recente é encaminhada ao modelo neural**.

Isso permite testar o modelo dentro do Pi, mas ainda não o transforma em um coding agent completo.

O checkpoint atual também não aprendeu o protocolo de tool calling. Ele pode responder texto dentro do Pi, porém ainda não sabe emitir chamadas estruturadas para editar arquivos, executar shell ou usar as ferramentas do Pi.

O próximo marco é treinar um checkpoint específico para Pi com:

1. contexto de pelo menos 512–2048 tokens;
2. exemplos de system prompt;
3. chamadas de ferramenta estruturadas;
4. resultados de ferramenta;
5. geração/correção de código Python;
6. benchmark de uso real de ferramentas.

## Fallback web

Com `--web-fallback`, o servidor mede a proporção de tokens desconhecidos no prompt para o tokenizer word-level. Acima do limiar (padrão 35%), ele consulta a web e retorna os resultados com título, snippet e URL.

A heurística é deliberadamente conservadora e ainda não é uma medida calibrada de confiança do LLM.
