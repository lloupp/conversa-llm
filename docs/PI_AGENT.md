# Conversa LLM no Pi Agent

O Conversa LLM expõe uma API local compatível com OpenAI Chat Completions, incluindo streaming e `tool_calls` nativo.

O Pi oferece por padrão `read`, `write`, `edit` e `bash`; também pode habilitar `powershell`, `grep`, `find` e `ls`. O adaptador do Conversa Pi aceita todas essas ferramentas.

## Instalação no Windows

```powershell
git clone https://github.com/lloupp/conversa-llm.git
cd conversa-llm
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[web]"
```

## Arquivos do checkpoint

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

Mescle `examples/pi/models.json` em:

```text
$HOME\.pi\agent\models.json
```

Abra o Pi, execute `/model` e selecione `Conversa Pi (local)`.

### Windows

O Pi usa Git Bash por padrão. Para expor PowerShell e as ferramentas de exploração, mescle `examples/pi/settings.windows.json` em `$HOME\.pi\agent\settings.json`.

## Contexto: transporte x modelo neural

O arquivo `models.json` anuncia 8192 tokens ao Pi para permitir que ele envie o envelope de sessão, schemas de ferramentas e histórico. O servidor não passa isso cru para a rede neural.

O checkpoint `cpu-8gb-pi` tem **512 tokens neurais**. Antes da inferência, o servidor:

1. descarta system prompts gigantes que não agregam ao pequeno modelo;
2. transforma os schemas das ferramentas em um catálogo compacto;
3. preserva os turnos recentes e resultados de ferramentas;
4. recorta o prompt final para caber no contexto neural.

Isso é uma forma de compaction local. Não significa que o modelo possua memória real de 8192 tokens.

## Ciclo de agente

O treino novo usa o mesmo formato renderizado que o servidor usa em produção:

```text
USUÁRIO: corrija o bug
ASSISTENTE: <tool_call>read...</tool_call>
RESULTADO read: ...
ASSISTENTE: <tool_call>edit...</tool_call>
RESULTADO edit: ...
ASSISTENTE: <tool_call>bash...</tool_call>
RESULTADO bash: FAILED...
ASSISTENTE: <tool_call>edit...</tool_call>
...
RESULTADO bash: passed
ASSISTENTE: correção concluída
```

Assim, a rede aprende não apenas a selecionar tools, mas também a reagir aos resultados.

## Roteador híbrido

Pedidos explícitos simples podem ser convertidos deterministicamente em chamadas para `read`, `write`, `edit`, `bash`, `powershell`, `grep`, `find` e `ls`. Isso é uma camada de confiabilidade; não deve ser confundido com inteligência neural.

Use `--no-hybrid-tools` ao medir o modelo sem essa ajuda.

## Busca web

Com `--web-fallback`, perguntas textuais fora do conhecimento local podem usar busca web. O fallback devolve resultados/fontes; não concede à rede neural conhecimento permanente.

## Segurança

As ferramentas do Pi rodam com as permissões normais do processo. Mantenha o projeto sob Git e revise operações destrutivas.

## Limite atual

Mesmo com alguns milhões de parâmetros, este continua sendo um modelo experimental treinado do zero em um PC doméstico. O objetivo é ganhar capacidade verificável em tarefas pequenas de engenharia de software; projetos grandes e raciocínio profundo ainda exigirão muito mais dados e computação.
