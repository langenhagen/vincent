# Vincent Voice Chat

Single-purpose voice chat CLI:
- records microphone audio until you press Enter,
- transcribes with local Whisper,
- sends each turn to `opencode run`,
- optionally speaks assistant replies with Kokoro (`--voice`).

## Setup

```bash
uv sync
```

Use Python 3.13 for best Kokoro compatibility.

## Usage

```bash
# Text only
uv run python voice_chat.py

# Enable assistant speech output
uv run python voice_chat.py --voice

# Useful options
uv run python voice_chat.py --new-session
uv run python voice_chat.py --session-id ses_abc123
uv run python voice_chat.py --task translate
uv run python voice_chat.py --model small
uv run python voice_chat.py --voice --tts-voice af_heart --tts-lang-code a --tts-speed 1.0
```

Say `exit` or `quit` to stop.
