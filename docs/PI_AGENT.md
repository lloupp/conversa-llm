# Conversa LLM no Pi Agent

O Conversa LLM expõe uma API local compatível com OpenAI Chat Completions:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- streaming SSE
- `tool_calls` nativo no formato OpenAI

## Instalação no Windows

```powershell
git clone https://github.com/lloupp/conversa-llm.git
cd conversa-llm
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[web]"
```

## Arquivos do checkpoint Pi

Coloque:

```text
checkpoints/conversa-pi.pt
data/pi_bpe.json
```

## Subir o servidor

```powershell
python -m conversa_llm.openai_server `
  --model checkpoints/conversa-pi.pt `
  --tokenizer-file data/pi_bpe.json `
  --model-id conversa-pi `
  --web-fallback
```

Teste:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

## Configurar o Pi

Copie ou mescle `examples/pi/models.json` em:

```text
$HOME\.pi\agent\models.json
```

Depois abra o Pi, execute `/model` e selecione:

```text
Conversa Pi (local)
```

### Windows: PowerShell como ferramenta

O Pi usa Git Bash por padrão no Windows. Se preferir PowerShell, mescle o conteúdo de:

```text
examples/pi/settings.windows.json
```

em:

```text
$HOME\.pi\agent\settings.json
```

## Como o tool calling funciona

O modelo pode emitir internamente:

```text
<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>
```

O adaptador converte isso para o formato OpenAI:

```json
{
  "tool_calls": [{
    "type": "function",
    "function": {
      "name": "read",
      "arguments": "{\"path\":\"README.md\"}"
    }
  }]
}
```

O Pi executa a ferramenta e devolve o resultado numa mensagem `tool`. O servidor inclui esse resultado no próximo contexto do modelo.

## Roteador híbrido

Para pedidos explícitos e simples, o adaptador não depende apenas da geração neural. Exemplos:

- "Leia src/app.py"
- "Execute no shell o comando `pytest -q`"
- "Substitua alpha por beta em src/app.py"

O roteador extrai a intenção e os argumentos diretamente e produz a chamada estruturada. Isso reduz erros de cópia de caminhos/comandos em um modelo pequeno.

Use `--no-hybrid-tools` para medir apenas o comportamento neural.

## Busca web

Com `--web-fallback`, perguntas textuais fora do conhecimento local podem cair no mecanismo de busca e retornar fontes/snippets.

O fallback web não substitui tool calling do Pi e não executa comandos externos.

## Segurança

As ferramentas do Pi operam com as permissões normais do processo. Use Git para rollback, revise comandos destrutivos e rode o Pi dentro de um diretório de projeto apropriado.

## Limites atuais

O modelo continua pequeno e não deve ser comparado a um LLM comercial:

- contexto neural: 256 tokens;
- bom para chamadas curtas de ferramenta e pequenas funções Python;
- histórico longo é compactado;
- tarefas complexas de arquitetura ainda exigem modelos maiores;
- o roteador híbrido cobre intenções explícitas, mas não raciocínio arbitrário de ferramenta.

O objetivo deste marco é ser funcional como agente local pequeno e treinável no seu PC de 8 GB.
