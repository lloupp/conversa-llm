from __future__ import annotations

import json
import random
import re
import unicodedata
from pathlib import Path


CONCEPTS = [
    ("variavel", ["Como atribuo 10 a x em Python?", "Qual sintaxe guarda o valor 10 em x?", "Mostre uma atribuição simples para x com valor 10."], "x = 10", "Como escrever em Python que x recebe dez?"),
    ("len", ["Como descubro o tamanho de uma lista?", "Qual função retorna quantos itens há em uma coleção?", "Como contar os elementos de [1, 2, 3]?"], "Use len(...).", "Qual função embutida informa o número de itens de uma sequência?"),
    ("type", ["Como descubro o tipo de um valor?", "Qual função mostra a classe de um objeto?", "Como verificar se 10 é int usando uma função de inspeção?"], "Use type(valor).", "Qual função retorna o tipo de um objeto?"),
    ("div_inteira", ["Qual operador faz divisão inteira?", "Como obter quociente inteiro em Python?", "Que operador usar em 7 por 2 para obter 3?"], "Use //.", "Qual operador retorna 3 em 7 dividido por 2 com parte fracionária descartada?"),
    ("modulo", ["Qual operador retorna o resto da divisão?", "Como testar resto em Python?", "Que operador usar para saber se n é par?"], "Use %.", "Qual operador calcula o resto de uma divisão?"),
    ("potencia", ["Qual operador calcula potência?", "Como escrever 2 elevado a 3 em Python?", "Que operador produz 8 com 2 e 3?"], "Use **.", "Qual operador representa exponenciação em Python?"),
    ("string_indice", ["Como pegar o primeiro caractere de texto?", "Qual índice acessa o primeiro item de uma string?", "Como acessar o início de s por índice?"], "Use texto[0].", "Como acessar o primeiro caractere da string s?"),
    ("slice", ["Como obter os três primeiros itens de uma lista?", "Como fatiar uma sequência do início até antes do índice 3?", "Qual slice pega índices 0, 1 e 2?"], "Use seq[:3].", "Qual expressão retorna os três primeiros elementos de seq?"),
    ("append", ["Como adicionar um item ao fim de uma lista?", "Qual método coloca um elemento no final da lista?", "Como inserir x no final de lista?"], "Use lista.append(x).", "Qual método adiciona x ao final de uma lista chamada lista?"),
    ("tuple", ["Tuplas podem ser alteradas depois de criadas?", "Uma tuple é mutável?", "Posso substituir um item de uma tupla diretamente?"], "Não. Tuplas são imutáveis.", "Uma tupla permite alterar seus elementos diretamente?"),
    ("dict", ["Como acessar a chave nome de um dicionário d?", "Qual sintaxe lê d na chave nome?", "Como recuperar o valor de nome em d usando colchetes?"], "Use d[\"nome\"].", "Como ler a chave \"nome\" do dicionário d?"),
    ("dict_get", ["Como buscar uma chave sem gerar KeyError se ela faltar?", "Qual método de dict aceita valor padrão?", "Como ler uma chave opcional de d?"], "Use d.get(chave, padrao).", "Qual método de dicionário permite fornecer um valor padrão quando a chave não existe?"),
    ("set", ["Qual coleção elimina duplicatas?", "Que tipo usar para itens únicos sem repetição?", "Como representar um conjunto de valores distintos?"], "Use set.", "Qual tipo embutido representa uma coleção de elementos únicos?"),
    ("range", ["range(3) produz quais números?", "Quais valores aparecem em list(range(3))?", "O limite final de range é incluído?"], "0, 1 e 2; o fim é exclusivo.", "Que valores list(range(3)) contém?"),
    ("for", ["Qual palavra-chave percorre itens de um iterável?", "Como iniciar um laço que visita cada item?", "Que comando é usado para iterar sobre uma lista?"], "Use for.", "Qual palavra-chave cria um laço sobre os itens de um iterável?"),
    ("while", ["Qual laço repete enquanto uma condição for verdadeira?", "Que palavra-chave cria repetição condicional?", "Como repetir enquanto x < 10?"], "Use while.", "Qual palavra-chave repete um bloco enquanto uma condição é verdadeira?"),
    ("break", ["Como sair imediatamente de um laço?", "Qual palavra-chave interrompe for ou while?", "Que comando encerra o loop atual?"], "Use break.", "Qual palavra-chave encerra imediatamente o laço atual?"),
    ("continue", ["Como pular para a próxima iteração do laço?", "Qual palavra-chave ignora o restante da iteração atual?", "Que comando avança o loop para a próxima volta?"], "Use continue.", "Qual palavra-chave pula o restante da iteração atual?"),
    ("funcao", ["Como declarar uma função chamada soma?", "Qual palavra-chave define funções?", "Como começar a definição de uma função em Python?"], "Use def.", "Qual palavra-chave inicia a definição de uma função?"),
    ("return", ["Como uma função devolve um valor?", "Qual palavra-chave retorna um resultado ao chamador?", "Que comando encerra a função entregando um valor?"], "Use return.", "Qual palavra-chave devolve um valor de uma função?"),
    ("default_arg", ["Como definir b com padrão 0 em def soma(a, b)?", "Qual sintaxe cria parâmetro padrão?", "Como fazer b ser opcional com valor zero?"], "Use def soma(a, b=0): ...", "Como declarar b com valor padrão 0 em uma função soma(a, b)?"),
    ("lambda", ["Qual palavra-chave cria uma função anônima curta?", "Como iniciar uma função anônima em Python?", "Que recurso cria funções pequenas sem def?"], "Use lambda.", "Qual palavra-chave cria uma função anônima?"),
    ("listcomp", ["Como criar quadrados de 0 a 2 com list comprehension?", "Dê uma compreensão de lista para x*x em range(3).", "Como gerar [0,1,4] por compreensão?"], "Use [x*x for x in range(3)].", "Qual compreensão de lista gera os quadrados dos valores em range(3)?"),
    ("enumerate", ["Como obter índice e valor ao iterar?", "Qual função combina posição e item num for?", "Como iterar recebendo índice e elemento?"], "Use enumerate(iteravel).", "Qual função embutida fornece pares de índice e valor durante a iteração?"),
    ("zip", ["Como percorrer duas listas em paralelo?", "Qual função agrupa elementos de iteráveis posição a posição?", "Como combinar a e b em pares durante um for?"], "Use zip(a, b).", "Qual função combina iteráveis elemento a elemento?"),
    ("try_except", ["Como capturar uma exceção em Python?", "Quais blocos básicos tratam erros esperados?", "Como tratar ValueError?"], "Use try e except.", "Quais palavras-chave formam o tratamento básico de exceções?"),
    ("finally", ["Qual bloco roda mesmo havendo ou não exceção?", "Onde colocar limpeza que deve sempre executar?", "Que cláusula de try executa sempre ao final?"], "Use finally.", "Qual cláusula de tratamento de exceções é executada independentemente de ocorrer erro?"),
    ("with", ["Qual palavra-chave gerencia contexto e fechamento de recursos?", "Como abrir arquivo garantindo fechamento automático?", "Que comando é usado com open para fechar o arquivo automaticamente?"], "Use with.", "Qual palavra-chave é usada com gerenciadores de contexto?"),
    ("class", ["Qual palavra-chave define uma classe?", "Como iniciar uma classe Pessoa?", "Que comando declara um novo tipo definido pelo usuário?"], "Use class.", "Qual palavra-chave inicia a definição de uma classe?"),
    ("self", ["Como se chama convencionalmente o primeiro parâmetro de método de instância?", "Qual nome costuma referenciar a própria instância?", "Em métodos de instância, qual parâmetro representa o objeto?"], "Convencionalmente, self.", "Qual nome convencional representa a instância atual dentro de um método?"),
    ("init", ["Qual método inicializa uma nova instância?", "Que método especial recebe os argumentos da construção do objeto?", "Qual método costuma atribuir atributos iniciais?"], "Use __init__.", "Qual método especial normalmente inicializa os atributos de uma instância?"),
    ("import", ["Qual palavra-chave carrega um módulo?", "Como disponibilizar nomes de outro módulo?", "Que comando básico importa math?"], "Use import.", "Qual palavra-chave importa um módulo?"),
    ("main", ["Para que serve if __name__ == '__main__'?", "Como executar um bloco só quando o arquivo é rodado diretamente?", "Qual guarda distingue execução direta de importação?"], "Use if __name__ == \"__main__\":.", "Qual guarda permite executar código apenas quando o módulo é iniciado diretamente?"),
    ("venv", ["Como criar um ambiente virtual com a biblioteca padrão?", "Qual comando cria uma venv chamada .venv?", "Como isolar dependências num ambiente virtual?"], "Use python -m venv .venv.", "Qual comando cria um ambiente virtual chamado .venv?"),
    ("pip", ["Como instalar requests usando o mesmo Python ativo?", "Qual comando recomendado chama pip como módulo?", "Como instalar um pacote com pip via python -m?"], "Use python -m pip install requests.", "Qual comando instala requests usando pip associado ao interpretador Python atual?"),
    ("fstring", ["Como interpolar nome numa string moderna?", "Qual prefixo habilita f-string?", "Como escrever uma string com {nome} avaliado?"], "Use f\"Olá, {nome}\".", "Qual sintaxe interpola a variável nome em uma f-string?"),
    ("strip", ["Qual método remove espaços das extremidades de uma string?", "Como tirar whitespace no começo e fim de texto?", "Que método limpa espaços externos?"], "Use texto.strip().", "Qual método remove whitespace das duas extremidades de uma string?"),
    ("split", ["Como dividir uma string em partes?", "Qual método separa texto usando um delimitador?", "Como transformar 'a,b' em partes usando vírgula?"], "Use texto.split(delimitador).", "Qual método de string divide o texto em uma lista de partes?"),
    ("join", ["Como juntar strings de uma lista com vírgula?", "Qual método concatena vários textos usando separador?", "Como unir itens por ', '?"], "Use \", \".join(itens).", "Qual expressão une as strings de itens usando vírgula e espaço?"),
    ("sorted", ["Como obter uma nova lista ordenada sem alterar a original?", "Qual função retorna os itens ordenados?", "Como ordenar um iterável produzindo novo resultado?"], "Use sorted(iteravel).", "Qual função embutida retorna uma nova sequência ordenada a partir de um iterável?"),
    ("list_sort", ["Qual método ordena uma lista no próprio objeto?", "Como ordenar uma lista in-place?", "Que método modifica a própria lista ao ordenar?"], "Use lista.sort().", "Qual método ordena uma lista in-place?"),
    ("is_vs_eq", ["Qual operador testa identidade de objetos?", "Para comparar se duas referências apontam ao mesmo objeto, uso qual operador?", "Que operador não deve ser confundido com igualdade ==?"], "Use is para identidade; == compara valores.", "Qual operador verifica identidade de objeto, em contraste com igualdade de valor?"),
    ("none", ["Qual valor representa ausência de valor?", "Que singleton costuma indicar nenhum resultado?", "Qual objeto especial representa 'sem valor'?"], "Use None.", "Qual objeto especial representa ausência de valor em Python?"),
    ("booleanos", ["Quais são os dois valores booleanos de Python?", "Como se escrevem verdadeiro e falso em Python?", "Quais literais pertencem a bool?"], "True e False.", "Quais são os dois literais booleanos de Python?"),
    ("and", ["Qual operador lógico exige que ambas as condições sejam verdadeiras?", "Como combinar duas condições com conjunção lógica?", "Qual operador representa E lógico?"], "Use and.", "Qual operador representa conjunção lógica em Python?"),
    ("or", ["Qual operador lógico aceita que uma das condições seja verdadeira?", "Como combinar condições com OU lógico?", "Qual operador representa disjunção?"], "Use or.", "Qual operador representa OU lógico em Python?"),
    ("not", ["Qual operador lógico nega uma condição?", "Como inverter um booleano?", "Que palavra-chave transforma verdadeiro em falso e vice-versa?"], "Use not.", "Qual operador lógico nega o valor de uma condição?"),
    ("any", ["Qual função retorna True se ao menos um item for verdadeiro?", "Como testar se existe algum valor truthy num iterável?", "Que função faz OU sobre um iterável?"], "Use any(iteravel).", "Qual função embutida retorna True quando pelo menos um elemento é truthy?"),
    ("all", ["Qual função retorna True se todos os itens forem verdadeiros?", "Como testar se todo valor de um iterável é truthy?", "Que função faz E sobre um iterável?"], "Use all(iteravel).", "Qual função embutida retorna True somente quando todos os elementos são truthy?"),
    ("copy_alias", ["Se b = a para uma lista, b é cópia independente?", "Atribuir uma lista a outro nome duplica os elementos?", "b = a cria nova lista?"], "Não. b e a referenciam a mesma lista.", "Depois de b = a, onde a é uma lista, b é uma cópia independente?"),
]


SHORT_TARGETS = {
    "variavel": "x = 10", "len": "len", "type": "type", "div_inteira": "//",
    "modulo": "%", "potencia": "**", "string_indice": "texto[0]", "slice": "seq[:3]",
    "append": "lista.append(x)", "tuple": "imutavel", "dict": "d[\"nome\"]",
    "dict_get": "d.get", "set": "set", "range": "0, 1, 2", "for": "for",
    "while": "while", "break": "break", "continue": "continue", "funcao": "def",
    "return": "return", "default_arg": "b=0", "lambda": "lambda",
    "listcomp": "[x*x for x in range(3)]", "enumerate": "enumerate", "zip": "zip",
    "try_except": "try e except", "finally": "finally", "with": "with", "class": "class",
    "self": "self", "init": "__init__", "import": "import",
    "main": "if __name__ == \"__main__\":", "venv": "python -m venv .venv",
    "pip": "python -m pip install requests", "fstring": "f\"Ola, {nome}\"",
    "strip": "strip", "split": "split", "join": "\", \".join(itens)",
    "sorted": "sorted", "list_sort": "sort", "is_vs_eq": "is", "none": "None",
    "booleanos": "True e False", "and": "and", "or": "or", "not": "not",
    "any": "any", "all": "all", "copy_alias": "nao",
}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


STOPWORDS = {
    "a", "ao", "as", "com", "como", "da", "das", "de", "do", "dos", "e", "em",
    "no", "na", "num", "o", "os", "ou", "para", "por", "python", "qual", "que", "se",
    "sem", "um", "uma", "usar", "use", "valor", "valores",
}


def content_words(text: str) -> list[str]:
    folded = unicodedata.normalize("NFKD", text.lower())
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    words = re.findall(r"[a-z_]+|//|\*\*|%", folded)
    return [word for word in words if word not in STOPWORDS and len(word) > 1 or word in {"//", "**", "%"}]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    training = []
    for _, variants, answer, _ in CONCEPTS:
        for question in variants:
            training.append({"messages": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]})

    answers = [answer for _, _, answer, _ in CONCEPTS]
    benchmark = []
    rng = random.Random(20260918)
    for idx, (skill, _, answer, heldout) in enumerate(CONCEPTS):
        distractors = [a for a in answers if a != answer]
        choices = rng.sample(distractors, 9) + [answer]
        rng.shuffle(choices)
        benchmark.append({"skill": skill, "question": heldout, "answer": answer, "choices": choices})

    recall_training = []
    generation_benchmark = []
    wrappers = [
        "{q}",
        "Python: {q}",
        "Responda: {q}",
        "Objetivamente: {q}",
        "Sem explicar: {q}",
        "Diga a sintaxe: {q}",
        "Python basico: {q}",
        "Resposta curta: {q}",
        "Qual e a resposta? {q}",
        "Em Python, {q}",
    ]
    for skill, variants, _, heldout in CONCEPTS:
        target = SHORT_TARGETS[skill]
        for question in variants:
            for wrapper in wrappers:
                recall_training.append({
                    "messages": [
                        {"role": "user", "content": wrapper.format(q=question)},
                        {"role": "assistant", "content": target},
                    ]
                })
            keywords = content_words(question)
            if keywords:
                recall_training.append({
                    "messages": [
                        {"role": "user", "content": "Conceito Python: " + " ".join(keywords)},
                        {"role": "assistant", "content": target},
                    ]
                })
                recall_training.append({
                    "messages": [
                        {"role": "user", "content": "Palavras-chave: " + " ".join(reversed(keywords))},
                        {"role": "assistant", "content": target},
                    ]
                })
        generation_benchmark.append({
            "skill": skill,
            "question": heldout + " Responda de forma curta.",
            "answer": target,
        })

    write_jsonl(root / "data" / "python_curriculum.jsonl", training)
    write_jsonl(root / "data" / "python_benchmark.jsonl", benchmark)
    write_jsonl(root / "data" / "python_recall_curriculum.jsonl", recall_training)
    write_jsonl(root / "data" / "python_generation_benchmark.jsonl", generation_benchmark)
    print(
        f"training={len(training)} benchmark={len(benchmark)} "
        f"recall={len(recall_training)} generation_benchmark={len(generation_benchmark)}"
    )


if __name__ == "__main__":
    main()
