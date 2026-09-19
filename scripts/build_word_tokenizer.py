from __future__ import annotations

import argparse

from conversa_llm.word_tokenizer import WordTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--out", default="data/python_word_vocab.json")
    parser.add_argument("--max-vocab", type=int, default=4096)
    args = parser.parse_args()
    tokenizer = WordTokenizer.train_from_jsonl(args.data, max_vocab=args.max_vocab)
    tokenizer.save(args.out)
    print(f"vocab={tokenizer.vocab_size} salvo={args.out}")


if __name__ == "__main__":
    main()
