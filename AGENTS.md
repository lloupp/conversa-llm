# Conversa LLM project instructions

- Keep the model trainable on an 8 GB RAM machine.
- Do not commit checkpoints (*.pt, *.pth) to Git.
- Run pytest after code changes.
- Preserve compatibility with the Python 58 checkpoint and the Pi BPE checkpoint.
- Keep tool execution delegated to Pi; the model server must never execute Pi tool calls itself.
