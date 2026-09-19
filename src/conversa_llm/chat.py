from __future__ import annotations

import argparse

import torch

from .config import ModelConfig
from .model import ConversaGPT
from .tokenizer import ByteTokenizer
from .word_tokenizer import WordTokenizer


def load_model(path: str, device: str) -> ConversaGPT:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = ModelConfig(**checkpoint["config"])
    model = ConversaGPT(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def answer(model: ConversaGPT, tokenizer: ByteTokenizer, prompt: str, device: str) -> str:
    prefix = tokenizer.encode_prompt(prompt)
    input_ids = torch.tensor([prefix], dtype=torch.long, device=device)
    output = model.generate(
        input_ids,
        max_new_tokens=160,
        temperature=0.7,
        top_k=30,
        stop_tokens={tokenizer.EOS, tokenizer.USER, tokenizer.SEP},
    )[0].tolist()
    generated = output[len(prefix) :]
    return tokenizer.decode_text(generated).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat local com o Conversa LLM.")
    parser.add_argument("--model", default="checkpoints/model.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tokenizer-file", default=None)
    args = parser.parse_args()

    tokenizer = WordTokenizer.load(args.tokenizer_file) if args.tokenizer_file else ByteTokenizer()
    model = load_model(args.model, args.device)
    print("Conversa LLM. Digite /sair para encerrar.")
    while True:
        prompt = input("Você> ").strip()
        if prompt.lower() in {"/sair", "/exit", "/quit"}:
            break
        print("LLM>", answer(model, tokenizer, prompt, args.device))


if __name__ == "__main__":
    main()
