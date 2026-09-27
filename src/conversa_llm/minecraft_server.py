from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .minecraft_decision import MinecraftDecisionRuntime


class Handler(BaseHTTPRequestHandler):
    runtime: MinecraftDecisionRuntime

    def log_message(self, fmt, *args):
        print("[minecraft-decision] " + (fmt % args))

    def send_json(self, status, payload):
        raw = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/health":
            return self.send_json(200, {"status": "ok", "profile": "minecraft"})
        self.send_json(404, {"error": "not_found"})

    def do_POST(self):
        if self.path != "/v1/minecraft/decision":
            return self.send_json(404, {"error": "not_found"})
        try:
            size = int(self.headers.get("content-length", "0"))
            body = json.loads(self.rfile.read(size) or b"{}")
            state = body.get("state")
            if isinstance(state, dict):
                state = json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if not isinstance(state, str) or not state.strip():
                raise ValueError("state obrigatório")
            self.send_json(200, self.runtime.decide_state(state))
        except Exception as exc:
            self.send_json(400, {"error": str(exc)})


def main():
    p = argparse.ArgumentParser(description="Servidor local do perfil Minecraft.")
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer-file", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threshold", type=float, default=0.70)
    args = p.parse_args()
    Handler.runtime = MinecraftDecisionRuntime(args.model, args.tokenizer_file, args.device, args.threshold)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"minecraft decision server http://{args.host}:{args.port}/v1/minecraft/decision")
    server.serve_forever()


if __name__ == "__main__":
    main()
