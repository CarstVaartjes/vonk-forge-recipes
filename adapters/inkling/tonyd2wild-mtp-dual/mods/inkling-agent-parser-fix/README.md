# Inkling agent parser fix

The tested vLLM build can expose the dedicated `<|end_message|>` special token
as visible content after an assistant tool call is followed by a tool result.
Normal one-turn chat and the outbound tool call still look correct, so a basic
smoke test does not catch the failure.

This mod patches the Inkling unified parser to remove only a terminal
`<|end_message|>` marker from parsed content in both streaming and non-streaming
responses. It does not register that token as an EOS token: Inkling also uses
it between its reasoning and final-answer blocks, so treating it as EOS would
truncate the answer after reasoning.

The patch is idempotent and checks the expected source anchors before editing.
Run `scripts/test-agent-tool-roundtrip.py` against the server after every
runtime/image update.

Version 2 also filters the independent non-streaming reasoning-adapter return
path. This matters because vLLM runs reasoning parsing before tool parsing; a
single-pass tool-parser patch alone does not clean a normal final answer.
