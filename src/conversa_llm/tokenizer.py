from __future__ import annotations


class ByteTokenizer:
    """Tokenizer UTF-8 determinístico, sem vocabulário pré-treinado.

    IDs 0..255 representam bytes. Os demais IDs são tokens de controle.
    """

    PAD = 256
    BOS = 257
    EOS = 258
    USER = 259
    ASSISTANT = 260
    SEP = 261
    vocab_size = 262

    def encode_text(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode_text(self, token_ids: list[int]) -> str:
        raw = bytes(token for token in token_ids if 0 <= token < 256)
        return raw.decode("utf-8", errors="ignore")

    def encode_prompt(self, text: str) -> list[int]:
        return [self.BOS, self.USER, *self.encode_text(text), self.SEP, self.ASSISTANT]

    def encode_conversation(self, messages: list[dict[str, str]]) -> tuple[list[int], list[bool]]:
        """Retorna tokens e máscara que marca conteúdo de respostas do assistente."""
        tokens: list[int] = [self.BOS]
        assistant_content_mask: list[bool] = [False]

        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "user":
                role_token = self.USER
                supervise = False
            elif role == "assistant":
                role_token = self.ASSISTANT
                supervise = True
            else:
                raise ValueError(f"role inválido: {role}")

            content_tokens = self.encode_text(content)
            tokens.append(role_token)
            assistant_content_mask.append(False)
            tokens.extend(content_tokens)
            assistant_content_mask.extend([supervise] * len(content_tokens))
            tokens.append(self.SEP)
            assistant_content_mask.append(supervise)

        tokens.append(self.EOS)
        assistant_content_mask.append(bool(messages and messages[-1]["role"] == "assistant"))
        return tokens, assistant_content_mask
