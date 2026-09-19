from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import torch

from .chat import load_model
from .tokenizer import ByteTokenizer
from .web_search import search_web
from .word_tokenizer import WordTokenizer


DEFAULT_MODEL_ID = "conversa-python-58"


def message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in {"text", "input_text"} and isinstance(part.get("text"), str):
                parts.append(part["text"])
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
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)] or [""]


class LocalBackend:
    def __init__(
        self,
        model_path: str,
        tokenizer_file: str | None,
        device: str,
        model_id: str = DEFAULT_MODEL_ID,
        web_fallback: bool = False,
        unknown_threshold: float = 0.35,
    ):
        self.model = load_model(model_path, device)
        self.tokenizer = (
            WordTokenizer.load(tokenizer_file) if tokenizer_file else ByteTokenizer()
        )
        self.device = device
        self.model_id = model_id
        self.web_fallback = web_fallback
        self.unknown_threshold = unknown_threshold
        self.lock = threading.Lock()

    @property
    def effective_context(self) -> int:
        return self.model.config.context_length

    def unknown_ratio(self, prompt: str) -> float:
        unk = getattr(self.tokenizer, "UNK", None)
        if unk is None:
            return 0.0
        ids = self.tokenizer.encode_text(prompt)
        if not ids:
            return 0.0
        return sum(token == unk for token in ids) / len(ids)

    def token_count(self, text: str) -> int:
        return len(self.tokenizer.encode_text(text))

    def answer(
        self,
        prompt: str,
        max_tokens: int = 48,
        temperature: float = 0.7,
        top_k: int = 30,
    ) -> str:
        if self.web_fallback and self.unknown_ratio(prompt) >= self.unknown_threshold:
            return search_web(prompt)

        prefix = self.tokenizer.encode_prompt(prompt)
        input_ids = torch.tensor([prefix], dtype=torch.long, device=self.device)
        max_tokens = max(1, min(int(max_tokens), 128))

        with self.lock, torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=max(float(temperature), 1e-5),
                top_k=max(1, int(top_k)),
                stop_tokens={
                    self.tokenizer.EOS,
                    self.tokenizer.USER,
                    self.tokenizer.SEP,
                },
            )[0].tolist()

        generated = output[len(prefix) :]
        answer = self.tokenizer.decode_text(generated).strip()
        if answer:
            return answer
        if self.web_fallback:
            return search_web(prompt)
        return "Não sei responder a isso ainda."


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
            self._send_json(
                200,
                {
                    "status": "ok",
                    "model": self.backend.model_id,
                    "effective_context": self.backend.effective_context,
                    "web_fallback": self.backend.web_fallback,
                },
            )
            return
        if self.path == "/v1/models":
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.backend.model_id,
                            "object": "model",
                            "created": 0,
                            "owned_by": "local",
                        }
                    ],
                },
            )
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
            prompt = latest_user_prompt(messages)
            max_tokens = body.get("max_tokens", body.get("max_completion_tokens", 48))
            temperature = body.get("temperature", 0.7)
            top_k = body.get("top_k", 30)
            reply = self.backend.answer(prompt, max_tokens, temperature, top_k)
        except Exception as exc:
            self._send_json(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
            return

        requested_model = body.get("model") or self.backend.model_id
        response_id = "chatcmpl-" + uuid.uuid4().hex
        created = int(time.time())

        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                first = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": requested_model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": ""},
                            "finish_reason": None,
                        }
                    ],
                }
                self.wfile.write(("data: " + json.dumps(first, ensure_ascii=False) + "\n\n").encode())
                self.wfile.flush()

                for part in split_stream_text(reply):
                    chunk = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": requested_model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": part},
                                "finish_reason": None,
                            }
                        ],
                    }
                    self.wfile.write(("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n").encode())
                    self.wfile.flush()

                final = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": requested_model,
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": "stop"}
                    ],
                }
                self.wfile.write(("data: " + json.dumps(final) + "\n\n").encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        prompt_tokens = self.backend.token_count(prompt)
        completion_tokens = self.backend.token_count(reply)
        self._send_json(
            200,
            {
                "id": response_id,
                "object": "chat.completion",
                "created": created,
                "model": requested_model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": reply},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            },
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Servidor OpenAI-compatible para usar o Conversa LLM no Pi Agent."
    )
    parser.add_argument("--model", required=True, help="Checkpoint .pt")
    parser.add_argument("--tokenizer-file", default=None)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--web-fallback", action="store_true")
    parser.add_argument("--unknown-threshold", type=float, default=0.35)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    backend = LocalBackend(
        model_path=args.model,
        tokenizer_file=args.tokenizer_file,
        device=args.device,
        model_id=args.model_id,
        web_fallback=args.web_fallback,
        unknown_threshold=args.unknown_threshold,
    )
    OpenAIHandler.backend = backend
    server = ThreadingHTTPServer((args.host, args.port), OpenAIHandler)
    print(
        f"Conversa LLM API em http://{args.host}:{args.port}/v1 "
        f"model={args.model_id} contexto_efetivo={backend.effective_context} "
        f"web_fallback={backend.web_fallback}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
