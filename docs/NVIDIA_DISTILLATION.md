# Conversa Pi + NVIDIA: destilação supervisionada no Colab

O experimento usa **respostas supervisionadas** da API NVIDIA como exemplos para o Transformer generativo e, separadamente, decisões para o System One. Não há transferência de pesos nem distilação por logits do professor. O aluno conserva arquitetura, tokenizer e interface de ferramentas do Conversa Pi.

## Licença, segurança e custo

**Atenção:** acesso a modelos pelo NVIDIA API Catalog **não** implica direito de usar saídas para treinar outro modelo. Verifique os termos da API **e a licença específica do professor**. Algumas licenças exigem autorização expressa. Consulte:

- https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-community-models-license/
- https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
- https://assets.ngc.nvidia.com/products/api-catalog/legal/NVIDIA%20API%20Trial%20Terms%20of%20Service.pdf

A flag \`--license-confirmed\` significa que **você verificou** que o uso é permitido; ela não concede permissão. Não inclua prompts com dados pessoais nem segredos. Coloque a chave só em \`NVIDIA_API_KEY\` (Secrets do Colab). Não inclua chaves no Git.

O script não executa tools, não envia pesos à NVIDIA, impõe limite de chamadas (10 por padrão) e registra tokens reportados. Custos, cotas e disponibilidade precisam ser verificados na conta. Execute primeiro \`--dry-run\`.

## Colab (caminho recomendado)

Abra \`notebooks/NVIDIA_Conversa_Distillation_Colab.ipynb\`. Use um runtime com GPU se for treinar (a **inferência do professor** ocorre remotamente). Informe o **ID exato** do modelo professor e confirme sua licença. O notebook gera currrículos originais, coleta exemplos, treina os componentes e mede alguns benchmarks. **Nenhuma inferência real ou treino foi executado na preparação deste PR.**

## CLI

\`\`\`bash
pip install -e '.[dev]'
python scripts/distill_nvidia.py --dry-run
# Configure NVIDIA_API_KEY no ambiente/gerenciador de segredos.
python scripts/distill_nvidia.py --teacher MODELO_VERIFICADO --license-confirmed --limit 10
\`\`\`

Arquivos produzidos em \`data/distilled_nvidia/\`:
- \`generation_train.jsonl\`: \`{"prompt":"...","completion":"..."}\`
- \`decision_train.jsonl\`: \`{"state":"...","label":"..."}\`
- \`manifest.jsonl\`: ID, professor, decisão de aceitação, uso de tokens, retomada.

Sementes idênticas são deduplicadas, registros \`split=eval\` não viram treino, decisões discordantes do gabarito são rejeitadas. Se trocar de professor, informe um novo diretório \`--out-dir\`.

## Treinar no projeto existente

\`\`\`bash
python scripts/build_pi_curriculum.py
python scripts/build_pi_agent_curriculum.py
python scripts/build_decision_curriculum.py
python scripts/build_bpe_tokenizer.py --data data/pi_curriculum.jsonl data/pi_agent_curriculum.jsonl data/distilled_nvidia/generation_train.jsonl --out data/pi_bpe.json --vocab-size 768
python -m conversa_llm.train --profile cpu-8gb-pi --tokenizer-file data/pi_bpe.json --data data/pi_curriculum.jsonl data/pi_agent_curriculum.jsonl data/distilled_nvidia/generation_train.jsonl --steps 100 --out checkpoints/conversa-nvidia-sft.pt
python -m conversa_llm.train_decision --tokenizer-file data/pi_bpe.json --data data/decision_curriculum.jsonl data/distilled_nvidia/decision_train.jsonl --steps 100 --out checkpoints/decision-nvidia-sft.pt
\`\`\`

Se nenhum exemplo foi aceito em determinada categoria, remova o respectivo arquivo do comando. **Para continuar os pesos de um checkpoint já treinado**, use \`--init-from\`, **com o mesmo tokenizer e a mesma arquitetura**. O treinamento abortará se a configuração diferir. O \`train_decision.py\` atual não possui \`--init-from\`, portanto cria um novo modelo de decisões e preserva o anterior.

## Avaliação e limitações

Use checkpoints de saída diferentes dos originais. Avalie com \`eval_tools\`, \`eval_agent\`, \`eval_code\` e \`eval_decision\`, conservando benchmarks que **não entraram nos dados de treino**. Compare acurácia, Brier, ECE, capacidade multi-turn e código executável com o baseline. Código gerado pelo professor não é verificado automaticamente: valide os exemplos antes de treinos maiores. Não interprete geração de dados como melhora comprovada do aluno.
