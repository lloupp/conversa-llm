from __future__ import annotations


def search_web(query: str, max_results: int = 3) -> str:
    """Busca textual opcional para fallback quando o modelo local não conhece o tema."""
    try:
        from ddgs import DDGS
    except ImportError:
        return (
            "Não sei responder com segurança pelo modelo local e o fallback web não está "
            "instalado. Instale com: pip install -e \".[web]\""
        )

    try:
        results = list(DDGS().text(query, max_results=max_results))
    except Exception as exc:  # rede/provedor externo pode falhar
        return f"Não sei responder com segurança e a busca web falhou: {exc}"

    if not results:
        return "Não sei responder com segurança e a busca web não retornou resultados."

    lines = ["Não sei com segurança pelo modelo local. Busquei na web:"]
    for index, item in enumerate(results, start=1):
        title = (item.get("title") or "Sem título").strip()
        body = (item.get("body") or "").strip()
        href = (item.get("href") or item.get("url") or "").strip()
        lines.append(f"{index}. {title}")
        if body:
            lines.append(f"   {body}")
        if href:
            lines.append(f"   {href}")
    return "\n".join(lines)
