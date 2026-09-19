from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import torch

from .agent_prompt import fit_prompt, render_agent_prompt
from .chat import load_model
from .tokenizer_loader import load_tokenizer
from .tool_protocol import ParsedToolCall, parse_tool_calls, strip_tool_calls
from .tool_router import choose_tool_call, route_explicit_tool
from .web_search import search_web

DEFAULT_MODEL_ID = "conversa-pi"


def message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in {"text", "input_text"}:
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def latest_user_prompt(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            text = message_text(message.get("content"))
            if text.strip():
                return text.strip()
    raise ValueError("nenhuma mensagem de usuário com texto foi recebida")


def split_stream_text(text: str, chunk_size: int = 24) -> list[str]:
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)] or [""]


def _last_role(messages: list[dict[str, Any]]) -> str | None:
    for message in reversed(messages):
        role = message.get("role")
        if role not in {"system", "developer"}:
            return role if isinstance(role, str) else None
    return None


@dataclass
class AgentReply:
    content: str = ""
    tool_call: ParsedToolCall | None = None


class LocalBackend:
    def __init__(
        self,
        model_path: str,
        tokenizer_file: str | None,
        device: str,
        model_id: str = DEFAULT_MODEL_ID,
        web_fallback: bool = False,
        unknown_threshold: float = 0.35,
        hybrid_tools: bool = True,
    ):
        self.model = load_model(model_path, device)
        self.tokenizer = load_tokenizer(tokenizer_file)
        self.device = device
        self.model_id = model_id
        self.web_fallback = web_fallback
        self.unknown_threshold = unknown_threshold
        self.hybrid_tools = hybrid_tools
        self.lock = threading.Lock()

    @property
    def effective_context(self) -> int:
        return self.model.config.context_length

    def unknown_ratio(self, prompt: str) -> float:
        unk = getattr(self.tokenizer, "UNK", None)
        if unk is None:
            return 0.0
        ids = self.tokenizer.encode_text(prompt)
        return 0.0 if not ids else sum(token == unk for token in ids) / len(ids)

    def token_count(self, text: str) -> int:
        return len(self.tokenizer.encode_text(text))

    def _generate(self, prompt: str, max_tokens: int, temperature: float, top_k: int) -> str:
        prefix = fit_prompt(self.tokenizer, prompt, self.model.config.context_length, max_tokens)
        x = torch.tensor([prefix], dtype=torch.long, device=self.device)
        max_tokens = max(1, min(int(max_tokens), max(1, self.model.config.context_length // 2)))
        with self.lock, torch.no_grad():
            output = self.model.generate(
                x,
                max_new_tokens=max_tokens,
                temperature=max(float(temperature), 1e-5),
                top_k=max(1, int(top_k)),
                stop_tokens={self.tokenizer.EOS, self.tokenizer.USER, self.tokenizer.SEP},
            )[0].tolist()
        return self.tokenizer.decode_text(output[len(prefix):]).strip()

    def respond(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 96,
        temperature: float = 0.3,
        top_k: int = 20,
    ) -> AgentReply:
        tools = tools or []
        user_prompt = latest_user_prompt(messages)

        if tools and self.hybrid_tools and _last_role(messages) == "user":
            explicit = route_explicit_tool(user_prompt, tools)
            if explicit:
                return AgentReply(tool_call=explicit)

        prompt = render_agent_prompt(messages, tools) if tools else user_prompt

        if not tools and self.web_fallback and self.unknown_ratio(user_prompt) >= self.unknown_threshold:
            return AgentReply(content=search_web(user_prompt))

        generated = self._generate(prompt, max_tokens, temperature, top_k)
        allowed = {
            (tool.get("function") or {}).get("name")
            for tool in tools
            if isinstance(tool, dict) and tool.get("type") == "function"
        }
        allowed.discard(None)
        model_calls = parse_tool_calls(generated, allowed_tools=set(allowed)) if tools else []
        content = strip_tool_calls(generated) if tools else generated
        chosen = None
        if tools and _last_role(messages) == "user":
            chosen = choose_tool_call(user_prompt, tools, model_calls)
        elif tools and not content and model_calls:
            chosen = model_calls[0]
        if chosen:
            return AgentReply(content=content, tool_call=chosen)

        if content:
            return AgentReply(content=content)
        if self.web_fallback and not tools:
            return AgentReply(content=search_web(user_prompt))
        return AgentReply(content="Não sei responder a isso ainda.")


class OpenAIHandler(BaseHTTPRequestHandler):
    backend: LocalBackend

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[pi-api] {self.address_string()} - {format % args}")

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, {
                "status": "ok",
                "model": self.backend.model_id,
                "effective_context": self.backend.effective_context,
                "web_fallback": self.backend.web_fallback,
                "hybrid_tools": self.backend.hybrid_tools,
            })
            return
        if self.path == "/v1/models":
            self._send_json(200, {"object": "list", "data": [{
                "id": self.backend.model_id,
                "object": "model",
                "created": 0,
                "owned_by": "local",
            }]})
            return
        self._send_json(404, {"error": {"message": "rota não encontrada"}})

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self._send_json(404, {"error": {"message": "rota não encontrada"}})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            messages = body.get("messages") or []
            tools = body.get("tools") or []
            reply = self.backend.respond(
                messages,
                tools,
                body.get("max_tokens", body.get("max_completion_tokens", 96)),
                body.get("temperature", 0.3),
                body.get("top_k", 20),
            )
        except Exception as exc:
            self._send_json(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
            return

        requested_model = body.get("model") or self.backend.model_id
        response_id = "chatcmpl-" + uuid.uuid4().hex
        created = int(time.time())
        call_id = "call_" + uuid.uuid4().hex[:16]
        finish_reason = "tool_calls" if reply.tool_call else "stop"

        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                self._sse({
                    "id": response_id, "object": "chat.completion.chunk", "created": created,
                    "model": requested_model,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                })
                if reply.tool_call:
                    self._sse({
                        "id": response_id, "object": "chat.completion.chunk", "created": created,
                        "model": requested_model,
                        "choices": [{"index": 0, "delta": {"tool_calls": [{
                            "index": 0, "id": call_id, "type": "function",
                            "function": {
                                "name": reply.tool_call.name,
                                "arguments": json.dumps(
                                    reply.tool_call.arguments,
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                ),
                            },
                        }]}, "finish_reason": None}],
                    })
                else:
                    for part in split_stream_text(reply.content):
                        self._sse({
                            "id": response_id, "object": "chat.completion.chunk", "created": created,
                            "model": requested_model,
                            "choices": [{
                                "index": 0, "delta": {"content": part}, "finish_reason": None
                            }],
                        })
                self._sse({
                    "id": response_id, "object": "chat.completion.chunk", "created": created,
                    "model": requested_model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                })
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        message: dict[str, Any] = {"role": "assistant", "content": reply.content or None}
        if reply.tool_call:
            message["tool_calls"] = [{
                "id": call_id,
                "type": "function",
                "function": {
                    "name": reply.tool_call.name,
                    "arguments": json.dumps(
                        reply.tool_call.arguments,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            }]
        prompt_text = latest_user_prompt(messages)
        prompt_tokens = self.backend.token_count(prompt_text)
        completion_text = reply.content or (
            json.dumps(reply.tool_call.arguments) if reply.tool_call else ""
        )
        completion_tokens = self.backend.token_count(completion_text)
        self._send_json(200, {
            "id": response_id,
            "object": "chat.completion",
            "created": created,
            "model": requested_model,
            "choices": [{
                "index": 0, "message": message, "finish_reason": finish_reason
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        })

    def _sse(self, payload: dict[str, Any]) -> None:
        self.wfile.write(
            ("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode("utf-8")
        )
        self.wfile.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Servidor OpenAI-compatible para Pi Agent.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer-file", default=None)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--web-fallback", action="store_true")
    parser.add_argument("--unknown-threshold", type=float, default=0.35)
    parser.add_argument(
        "--no-hybrid-tools", action="store_true",
        help="desativa reparo/roteamento explícito",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    backend = LocalBackend(
        args.model, args.tokenizer_file, args.device, args.model_id,
        args.web_fallback, args.unknown_threshold,
        hybrid_tools=not args.no_hybrid_tools,
    )
    OpenAIHandler.backend = backend
    server = ThreadingHTTPServer((args.host, args.port), OpenAIHandler)
    print(
        f"Conversa LLM API em http://{args.host}:{args.port}/v1 "
        f"model={args.model_id} contexto={backend.effective_context} "
        f"web_fallback={backend.web_fallback} hybrid_tools={backend.hybrid_tools}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
