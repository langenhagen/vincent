"""Main Vincent CLI orchestration.

This module wires the full chat loop together: parse command-line options,
capture and transcribe microphone turns, send prompts to opencode, render
terminal output, and optionally speak assistant replies with Kokoro.
"""

# pylint: disable=import-error

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import TextIO

import sounddevice as sd

from .kokoro_output import KokoroSpeaker
from .opencode_client import OpenCodeRunOptions, ask_opencode
from .whisper_input import build_whisper_model, capture_turn

EXIT_PHRASES = {"exit", "quit", "goodbye"}
ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
SYSTEM_TEXT_COLOR = "\033[90m"
USER_LABEL_COLOR = "\033[34m"
ASSISTANT_LABEL_COLOR = "\033[32m"
ASSISTANT_TEXT_COLOR = "\033[36m"


def positive_int(value: str) -> int:
    """Parse an integer CLI value and ensure it is positive."""
    parsed = int(value)
    if parsed <= 0:
        msg = f"Expected a positive integer, got {value}"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def parse_args() -> argparse.Namespace:
    """Parse command-line flags for microphone-to-opencode voice chat."""
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Record microphone audio, transcribe with Whisper, and send turns "
            "to a long-lived opencode session."
        ),
    )

    parser.add_argument(
        "--whisper-model",
        default="base",
        help="Whisper model name/path",
    )
    parser.add_argument(
        "--whisper-device",
        default="auto",
        help="Whisper device: auto, cpu, cuda",
    )
    parser.add_argument(
        "--whisper-compute-type",
        default="int8",
        help="faster-whisper compute type (int8, float16, float32, ...)",
    )
    parser.add_argument(
        "--whisper-task",
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Whisper task: transcribe original language or translate to English",
    )
    parser.add_argument(
        "--input-language",
        default=None,
        help="Expected spoken language code, e.g. en, de, fr (omit for auto)",
    )
    parser.add_argument(
        "--input-sample-rate",
        type=positive_int,
        default=16000,
        help="Microphone input sample rate in Hz",
    )
    parser.add_argument(
        "--input-channels",
        type=positive_int,
        default=1,
        help="Microphone input channel count (1=mono, 2=stereo)",
    )
    parser.add_argument(
        "--keep-input-audio",
        action="store_true",
        help="Keep each recorded input WAV in .voice_inputs/<session>/",
    )

    session_group = parser.add_mutually_exclusive_group()
    session_group.add_argument(
        "--session-id",
        default=None,
        help="Reuse an existing opencode session id",
    )
    session_group.add_argument(
        "--new-session",
        action="store_true",
        help="Ignore any saved session and start a new one",
    )

    parser.add_argument(
        "--session-file",
        type=Path,
        default=Path(".voice_chat_state.json"),
        help="Path to store the current opencode session id",
    )
    parser.add_argument(
        "--opencode-model",
        default=None,
        help="Optional opencode model in provider/model format",
    )
    parser.add_argument(
        "--opencode-agent",
        default=None,
        help="Optional opencode agent",
    )
    parser.add_argument(
        "--opencode-attach",
        default=None,
        help="Optional opencode server URL, e.g. http://127.0.0.1:4096",
    )
    parser.add_argument(
        "--opencode-dir",
        default=None,
        help="Optional working directory for opencode run",
    )
    parser.add_argument(
        "--voice",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Speak assistant replies with Kokoro text-to-speech (enabled by default)",
    )
    parser.add_argument(
        "--tts-voice",
        default="af_heart",
        help="Kokoro voice id, e.g. af_heart",
    )
    parser.add_argument(
        "--tts-lang-code",
        default="a",
        help="Kokoro language code, usually 'a' (US) or 'b' (UK)",
    )
    parser.add_argument(
        "--tts-speed",
        type=float,
        default=1.0,
        help="Kokoro playback speed",
    )
    return parser.parse_args()


def stdout(message: str) -> None:
    """Write a message to standard output and flush immediately."""
    sys.stdout.write(message)
    sys.stdout.flush()


def stderr(message: str) -> None:
    """Write a message to standard error and flush immediately."""
    if supports_ansi(sys.stderr):
        message = f"{SYSTEM_TEXT_COLOR}{message}{ANSI_RESET}"
    sys.stderr.write(message)
    sys.stderr.flush()


def supports_ansi(stream: TextIO | None = None) -> bool:
    """Return True when terminal color output should be enabled."""
    if os.getenv("NO_COLOR") is not None:
        return False
    output_stream = stream or sys.stdout
    return output_stream.isatty()


def apply_ansi(text: str, *styles: str) -> str:
    """Wrap text in ANSI styles when terminal output supports it."""
    if not supports_ansi():
        return text

    prefix = "".join(styles)
    if not prefix:
        return text
    return f"{prefix}{text}{ANSI_RESET}"


def format_assistant_text(text: str) -> str:
    """Apply fixed ANSI style to assistant output text."""
    return apply_ansi(text, ASSISTANT_TEXT_COLOR)


def format_user_text(text: str) -> str:
    """Return user text unchanged."""
    return text


def format_user_label(text: str) -> str:
    """Apply fixed ANSI style to the user speaker label."""
    return apply_ansi(text, ANSI_BOLD, USER_LABEL_COLOR)


def format_assistant_label(text: str) -> str:
    """Apply fixed ANSI style to the assistant speaker label."""
    return apply_ansi(text, ANSI_BOLD, ASSISTANT_LABEL_COLOR)


def load_session_id(state_path: Path) -> str | None:
    """Load the stored opencode session id from disk."""
    if not state_path.exists():
        return None

    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        stderr(f"Could not read session file {state_path}: {exc}\n")
        return None

    session_id = data.get("session_id")
    if isinstance(session_id, str) and session_id.strip():
        return session_id.strip()
    return None


def save_session_id(state_path: Path, session_id: str) -> None:
    """Persist the active opencode session id to disk."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"session_id": session_id}
    state_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def resolve_session_id(args: argparse.Namespace, state_path: Path) -> str | None:
    """Pick the initial session id from CLI args and stored state."""
    if args.session_id:
        try:
            save_session_id(state_path, args.session_id)
        except OSError as exc:
            stderr(
                f"Could not persist session id to {state_path}: {exc}. "
                "Continuing with in-memory session only.\n",
            )
        return str(args.session_id)
    if args.new_session:
        return None
    return load_session_id(state_path)


# pylint: disable=too-many-branches,too-many-statements
def run_voice_chat(args: argparse.Namespace) -> None:  # noqa: C901, PLR0912, PLR0915
    """Run the continuous record/transcribe/ask/reply loop."""
    state_path = args.session_file.expanduser().resolve()
    session_id = resolve_session_id(args, state_path)
    try:
        whisper_model = build_whisper_model(args)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        stderr(f"Failed to load Whisper model: {exc}\n")
        raise SystemExit(2) from exc

    speaker: KokoroSpeaker | None = None
    if args.voice:
        try:
            speaker = KokoroSpeaker(
                lang_code=args.tts_lang_code,
                voice=args.tts_voice,
                speed=args.tts_speed,
            )
            stderr(
                "Kokoro TTS enabled "
                f"(voice={args.tts_voice}, "
                f"lang={args.tts_lang_code}, "
                f"speed={args.tts_speed}).\n",
            )
        except RuntimeError as exc:
            stderr(f"Voice requested but unavailable: {exc}\n")
            stderr("Run without --voice, or use Python 3.12/3.13 for Kokoro.\n")
            raise SystemExit(2) from exc

    if session_id:
        stderr(f"Using opencode session: {session_id}\n")
    else:
        stderr(
            "No saved opencode session found. A new session will be created on "
            "the first prompt.\n",
        )

    stderr("Speak, then press Enter to finish each turn.\n")
    stderr("Say 'exit' or 'quit' to end the loop.\n")

    while True:
        try:
            current_session = session_id or "new-session"
            user_text, detected_language = capture_turn(
                args,
                current_session,
                whisper_model,
                stderr,
            )
        except RuntimeError as exc:
            stderr(f"{exc}\n")
            stderr("Please try again.\n")
            continue
        except KeyboardInterrupt:
            stderr("Stopped.\n")
            return

        if not user_text:
            stderr("No speech detected.\n")
            continue

        styled_user = format_user_text(user_text)
        styled_user_label = format_user_label("You:")
        stdout(f"{styled_user_label}\n{styled_user}\n\n")
        if detected_language:
            stderr(f"Detected language: {detected_language}\n")

        if user_text.strip().lower() in EXIT_PHRASES:
            stderr("Exit phrase detected.\n")
            return

        stderr("Asking opencode...\n")
        try:
            opencode_options = OpenCodeRunOptions(
                session_id=session_id,
                model=args.opencode_model,
                agent=args.opencode_agent,
                attach=args.opencode_attach,
                directory=args.opencode_dir,
            )
            assistant_text, discovered_session_id = ask_opencode(
                prompt=user_text,
                options=opencode_options,
            )
        except RuntimeError as exc:
            stderr(f"{exc}\n")
            continue
        except KeyboardInterrupt:
            stderr("Stopped.\n")
            return

        if discovered_session_id and discovered_session_id != session_id:
            session_id = discovered_session_id
            try:
                save_session_id(state_path, session_id)
                stderr(f"Saved opencode session: {session_id} ({state_path})\n")
            except OSError as exc:
                stderr(
                    f"Could not persist discovered session id to {state_path}: {exc}\n",
                )

        if not assistant_text:
            stderr("opencode returned no text response.\n")
            continue

        styled = format_assistant_text(assistant_text)
        styled_assistant_label = format_assistant_label("Assistant:")
        stdout(f"{styled_assistant_label}\n{styled}\n\n")
        if speaker:
            try:
                speaker.speak(assistant_text)
            except KeyboardInterrupt:
                stderr("Stopped.\n")
                return
            except (RuntimeError, ValueError, OSError, sd.PortAudioError) as exc:
                stderr(f"Kokoro playback failed: {exc}\n")


def main() -> None:
    """Entry point for voice chat."""
    args = parse_args()
    run_voice_chat(args)


if __name__ == "__main__":
    main()
