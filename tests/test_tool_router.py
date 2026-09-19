from conversa_llm.pi_tools import PI_CORE_TOOLS as TOOLS
from conversa_llm.tool_router import route_explicit_tool


def test_routes_read_exact_path():
    call = route_explicit_tool(
        "Por favor, abra e leia src/router.py antes de continuar.", TOOLS
    )
    assert call is not None
    assert call.name == "read"
    assert call.arguments == {"path": "src/router.py"}


def test_routes_bash_exact_command():
    call = route_explicit_tool(
        "Execute no shell o comando `pytest tests -q` agora.", TOOLS
    )
    assert call is not None
    assert call.name == "bash"
    assert call.arguments == {"command": "pytest tests -q"}


def test_routes_edit_with_current_pi_shape():
    call = route_explicit_tool(
        "Substitua alpha por beta dentro de src/router.py.", TOOLS
    )
    assert call is not None
    assert call.name == "edit"
    assert call.arguments == {
        "path": "src/router.py",
        "edits": [{"oldText": "alpha", "newText": "beta"}],
    }


def test_does_not_infer_destructive_action_without_explicit_verb():
    assert route_explicit_tool("O arquivo src/app.py talvez precise mudar.", TOOLS) is None
