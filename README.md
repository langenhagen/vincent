# Vincent Voice Chat
A Whisper-, Kokoro- and Opencode-powered voice chat Agent CLI.

![](res/vincent.webp)

Voice chat with your AI agent:
- records microphone audio until you press Enter,
- transcribes with local Whisper,
- sends each turn to `opencode run`,
- speaks assistant replies with Kokoro.

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

## Troubleshooting

- `No module named pip`
  - Recreate the venv with seeding: `uv venv --python 3.13 --seed && uv sync`
  - Quick repair in existing env: `uv run python -m ensurepip --upgrade`

- `Warning: You are sending unauthenticated requests to the HF Hub`
  - Optional only. Set `HF_TOKEN` if you want better download limits:
    `export HF_TOKEN=hf_your_token_here`

- No audio output from Kokoro
  - Check detected devices:
    `uv run python -c "import sounddevice as sd; print(sd.default.device); print(sd.query_devices())"`
  - Verify with text-only mode to isolate TTS/device issues: `uv run vincent --no-voice`

## Usage

```bash
uv run vincent
uv run vincent --no-voice  # Disable assistant speech output

# Useful options
uv run vincent --help
uv run vincent --new-session
uv run vincent --session-id ses_abc123
uv run vincent --whisper-task translate
uv run vincent --whisper-model small
uv run vincent --input-language en
uv run vincent --input-sample-rate 16000 --input-channels 1
uv run vincent --keep-input-audio
uv run vincent --tts-voice af_heart --tts-lang-code a --tts-speed 1.0

# Look up Kokoro Language Codes, Voices and Whatnot
uv run kokoro-info --lang-codes
uv run kokoro-info --aliases
uv run kokoro-info --voices
```

Say `exit` or `quit` to stop.

## Input Flags

- `--input-language`: hint the spoken language for Whisper (faster and often more accurate); omit to auto-detect.
- `--input-sample-rate`: microphone capture rate in Hz (default `16000`, good for speech).
- `--input-channels`: microphone channel count (`1` mono is typical; `2` stereo if needed).
- `--keep-input-audio`: keep each turn's WAV in `.voice_inputs/<session>/`.
