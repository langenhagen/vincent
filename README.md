# Vincent Voice Chat
A Whisper-, Kokoro- and Opencode-powered voice chat Agent CLI.

Single-purpose voice chat CLI:
- records microphone audio until you press Enter,
- transcribes with local Whisper,
- sends each turn to `opencode run`,
- optionally speaks assistant replies with Kokoro (`--voice`).

## Setup

```bash
uv venv --python 3.13 --seed
uv sync
```

Use Python 3.13 for best Kokoro compatibility.

Why `--seed`? It installs base tooling like `pip` into `.venv`. This avoids
`No module named pip` errors that can appear during Kokoro initialization.

## Usage

```bash
uv run vincent  # Text output only
uv run vincent --voice  # Enable assistant speech output

# Useful options
uv run vincent --help
uv run vincent --new-session
uv run vincent --session-id ses_abc123
uv run vincent --task translate
uv run vincent --model small
uv run vincent --voice --tts-voice af_heart --tts-lang-code a --tts-speed 1.0
```

Say `exit` or `quit` to stop.
