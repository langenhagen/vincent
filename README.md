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

Note: `uv sync` also installs `en-core-web-sm` (spaCy English model) because
Kokoro's English voice pipeline depends on it.

Optional: set `HF_TOKEN` to avoid Hugging Face anonymous-request warnings and
get better download rate limits for Kokoro model assets.

```bash
export HF_TOKEN=hf_your_token_here
```

## Usage

```bash
uv run vincent  # Text output only
uv run vincent --voice  # Enable assistant speech output

# Useful options
uv run vincent --help
uv run vincent --new-session
uv run vincent --session-id ses_abc123
uv run vincent --whisper-task translate
uv run vincent --whisper-model small
uv run vincent --input-language en
uv run vincent --input-sample-rate 16000 --input-channels 1
uv run vincent --keep-input-audio
uv run vincent --voice --tts-voice af_heart --tts-lang-code a --tts-speed 1.0

# Look up Kokoro Language Codes, Voices and Whatnot
uv run vincent-kokoro-info --lang-codes
uv run vincent-kokoro-info --aliases
uv run vincent-kokoro-info --voices
```

Say `exit` or `quit` to stop.

## Input Flags

- `--input-language`: hint the spoken language for Whisper (faster and often more accurate); omit to auto-detect.
- `--input-sample-rate`: microphone capture rate in Hz (default `16000`, good for speech).
- `--input-channels`: microphone channel count (`1` mono is typical; `2` stereo if needed).
- `--keep-input-audio`: keep each turn's temporary WAV on disk for debugging.
