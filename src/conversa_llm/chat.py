from __future__ import annotations

import argparse

import torch

from .config import ModelConfig
from .model import ConversaGPT
from .tokenizer_loader import load_tokenizer


def load_model(path: str, device: str) -> ConversaGPT:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = ModelConfig(**checkpoint["config"])
    model = ConversaGPT(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def answer(model: ConversaGPT, tokenizer, prompt: str, device: str) -> str:
    prefix = tokenizer.encode_prompt(prompt)
    if len(prefix) >= model.config.context_length:
        prefix = prefix[:2] + prefix[-(model.config.context_length - 3):]
    input_ids = torch.tensor([prefix], dtype=torch.long, device=device)
    output = model.generate(
        input_ids,
        max_new_tokens=min(160, max(1, model.config.context_length // 2)),
        temperature=0.7,
        top_k=30,
        stop_tokens={tokenizer.EOS, tokenizer.USER, tokenizer.SEP},
    )[0].tolist()
    generated = output[len(prefix):]
    return tokenizer.decode_text(generated).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat local com o Conversa LLM.")
    parser.add_argument("--model", default="checkpoints/model.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tokenizer-file", default=None)
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer_file)
    model = load_model(args.model, args.device)
    print("Conversa LLM. Digite /sair para encerrar.")
    while True:
        prompt = input("Você> ").strip()
        if prompt.lower() in {"/sair", "/exit", "/quit"}:
            break
        print("LLM>", answer(model, tokenizer, prompt, args.device))


if __name__ == "__main__":
    main()
