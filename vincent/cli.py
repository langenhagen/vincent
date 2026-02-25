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
import subprocess  # nosec B404  # B404: required for opencode CLI subprocess call.
import sys
from pathlib import Path

import sounddevice as sd

from .kokoro_output import KokoroSpeaker
from .whisper_input import build_whisper_model, capture_turn

EXIT_PHRASES = {"exit", "quit", "goodbye"}
ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
SYSTEM_TEXT_COLOR = "\033[90m"
USER_LABEL_COLOR = "\033[34m"
ASSISTANT_LABEL_COLOR = "\033[32m"
ASSISTANT_TEXT_COLOR = "\033[36m"


def parse_args() -> argparse.Namespace:
    """Parse command-line flags for microphone-to-opencode voice chat."""
    parser = argparse.ArgumentParser(
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
        type=int,
        default=16000,
        help="Microphone input sample rate in Hz",
    )
    parser.add_argument(
        "--input-channels",
        type=int,
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
    if supports_ansi():
        message = f"{SYSTEM_TEXT_COLOR}{message}{ANSI_RESET}"
    sys.stderr.write(message)
    sys.stderr.flush()


def supports_ansi() -> bool:
    """Return True when terminal color output should be enabled."""
    if os.getenv("NO_COLOR") is not None:
        return False
    return sys.stdout.isatty()


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
        save_session_id(state_path, args.session_id)
        return str(args.session_id)
    if args.new_session:
        return None
    return load_session_id(state_path)


def build_opencode_command(
    message: str,
    args: argparse.Namespace,
    session_id: str | None,
) -> list[str]:
    """Build the opencode CLI command for one conversation turn."""
    command = ["opencode", "run", "--format", "json"]
    if session_id:
        command.extend(["--session", session_id])
    if args.opencode_model:
        command.extend(["--model", args.opencode_model])
    if args.opencode_agent:
        command.extend(["--agent", args.opencode_agent])
    if args.opencode_attach:
        command.extend(["--attach", args.opencode_attach])
    if args.opencode_dir:
        command.extend(["--dir", args.opencode_dir])
    command.append(message)
    return command


def parse_opencode_events(output: str) -> tuple[str, str | None]:
    """Parse JSON event lines and return response text plus session id."""
    response_chunks: list[str] = []
    discovered_session: str | None = None

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        event_session_id = event.get("sessionID")
        if isinstance(event_session_id, str) and event_session_id:
            discovered_session = event_session_id

        if event.get("type") != "text":
            continue

        part = event.get("part")
        if not isinstance(part, dict):
            continue

        text = part.get("text")
        if isinstance(text, str) and text:
            response_chunks.append(text)

    return "".join(response_chunks).strip(), discovered_session


def ask_opencode(
    prompt: str,
    args: argparse.Namespace,
    session_id: str | None,
) -> tuple[str, str | None]:
    """Send one prompt to opencode and return the response and session id."""
    command = build_opencode_command(prompt, args, session_id)
    # Fixed argv list; shell execution is explicitly disabled.
    result = subprocess.run(  # noqa: S603  # nosec B603
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip()
        msg = f"opencode run failed ({result.returncode}): {details}"
        raise RuntimeError(msg)

    response_text, discovered_session = parse_opencode_events(result.stdout)
    return response_text, discovered_session or session_id


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
            assistant_text, discovered_session_id = ask_opencode(
                prompt=user_text,
                args=args,
                session_id=session_id,
            )
        except RuntimeError as exc:
            stderr(f"{exc}\n")
            continue
        except KeyboardInterrupt:
            stderr("Stopped.\n")
            return

        if discovered_session_id and discovered_session_id != session_id:
            session_id = discovered_session_id
            save_session_id(state_path, session_id)
            stderr(f"Saved opencode session: {session_id} ({state_path})\n")

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
