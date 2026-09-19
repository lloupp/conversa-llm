from __future__ import annotations

import argparse

from conversa_llm.bpe_tokenizer import ByteBPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina tokenizer byte-BPE do zero.")
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--out", default="data/pi_bpe.json")
    parser.add_argument("--vocab-size", type=int, default=768)
    parser.add_argument("--min-pair-freq", type=int, default=2)
    args = parser.parse_args()
    tokenizer = ByteBPETokenizer.train_from_jsonl(
        args.data,
        vocab_size=args.vocab_size,
        min_pair_freq=args.min_pair_freq,
    )
    tokenizer.save(args.out)
    print(f"vocab={tokenizer.vocab_size} merges={len(tokenizer.merges)} salvo={args.out}")


if __name__ == "__main__":
    main()
