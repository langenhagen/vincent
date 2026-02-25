"""Voice chat bridge: microphone Whisper transcription + opencode session."""

# pylint: disable=import-error

import argparse
import contextlib
import importlib
import json
import os
import re
import subprocess  # nosec B404  # B404: required for opencode CLI subprocess call.
import sys
import tempfile
import threading
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from scipy.io.wavfile import write as wav_write

EXIT_PHRASES = {"exit", "quit", "goodbye"}
KEPT_INPUT_AUDIO_DIR = Path(".voice_inputs")
ANSI_RESET = "\033[0m"
ASSISTANT_COLOR_CODES = {
    "none": "",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "magenta": "\033[35m",
}
USER_COLOR_CODES = {
    "none": "",
    "blue": "\033[34m",
    "white": "\033[37m",
    "bright-black": "\033[90m",
    "green": "\033[32m",
    "yellow": "\033[33m",
}
SYSTEM_COLOR_CODES = {
    "none": "",
    "blue": "\033[34m",
    "white": "\033[37m",
    "bright-black": "\033[90m",
    "green": "\033[32m",
    "yellow": "\033[33m",
}
SYSTEM_COLOR_NAME = "bright-black"
SYSTEM_BOLD_ENABLED = False
USER_LABEL_COLOR = "\033[34m"
ASSISTANT_LABEL_COLOR = "\033[32m"
USER_TEXT_COLOR_NAME = "none"
USER_TEXT_BOLD_ENABLED = False
ASSISTANT_TEXT_COLOR_NAME = "cyan"
ASSISTANT_TEXT_BOLD_ENABLED = False


def whisper_to_text(
    wav_path: Path,
    args: argparse.Namespace,
) -> tuple[str, str | None]:
    """Run Whisper on a WAV file and return text plus detected language."""
    model = WhisperModel(
        args.whisper_model,
        device=args.whisper_device,
        compute_type=args.whisper_compute_type,
    )
    segments, info = model.transcribe(
        str(wav_path),
        task=args.whisper_task,
        language=args.input_language,
        vad_filter=True,
    )
    text = " ".join(
        segment.text.strip() for segment in segments if segment.text.strip()
    )
    return text.strip(), getattr(info, "language", None)


class KokoroSpeaker:  # pylint: disable=too-few-public-methods
    """Generate and play speech audio from assistant text with Kokoro."""

    def __init__(
        self,
        lang_code: str,
        voice: str,
        speed: float,
    ) -> None:
        """Initialize Kokoro pipeline and playback parameters."""
        warnings.filterwarnings(
            "ignore",
            message=("dropout option adds dropout after all but last recurrent layer"),
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=(
                "`torch.nn.utils.weight_norm` is deprecated in favor of "
                "`torch.nn.utils.parametrizations.weight_norm`"
            ),
            category=FutureWarning,
        )

        try:
            kokoro_module = importlib.import_module("kokoro")
        except Exception as exc:  # pylint: disable=broad-exception-caught
            msg = (
                "Kokoro could not be imported. Install with `uv add kokoro` and "
                "use Python 3.10-3.13 for voice mode."
            )
            raise RuntimeError(msg) from exc

        kpipeline = getattr(kokoro_module, "KPipeline", None)
        if kpipeline is None:
            msg = "Installed kokoro package does not expose KPipeline"
            raise RuntimeError(msg)

        try:
            self._pipeline = kpipeline(
                lang_code=lang_code,
                repo_id="hexgrad/Kokoro-82M",
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            msg = (
                "Kokoro failed to initialize. This is often a Python-version "
                "compatibility problem (Kokoro stack currently targets Python "
                "3.10-3.13, while this project uses 3.14)."
            )
            raise RuntimeError(msg) from exc
        self._voice = voice
        self._speed = speed
        self._sample_rate = 24000

    def speak(self, text: str) -> None:
        """Convert text to speech and play it through the default audio output."""
        generator = self._pipeline(
            text,
            voice=self._voice,
            speed=self._speed,
            split_pattern=r"\n+",
        )

        chunks = [audio for _, _, audio in generator if len(audio)]

        if not chunks:
            return

        output_audio = np.concatenate(chunks)
        sd.play(output_audio, samplerate=self._sample_rate)
        sd.wait()


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
        help=(
            "Keep each recorded input WAV in .voice_inputs/<session>/ with a "
            "YYYY-MM-DD-HH-MM-SS filename prefix"
        ),
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
        color = SYSTEM_COLOR_CODES.get(SYSTEM_COLOR_NAME, "")
        if SYSTEM_BOLD_ENABLED:
            color = f"\033[1m{color}"
        if color:
            message = f"{color}{message}{ANSI_RESET}"
    sys.stderr.write(message)
    sys.stderr.flush()


def supports_ansi() -> bool:
    """Return True when terminal color output should be enabled."""
    if os.getenv("NO_COLOR") is not None:
        return False
    return sys.stdout.isatty()


def format_assistant_text(text: str) -> str:
    """Apply optional ANSI style to assistant output text."""
    if not supports_ansi():
        return text

    color = ASSISTANT_COLOR_CODES.get(ASSISTANT_TEXT_COLOR_NAME, "")
    if ASSISTANT_TEXT_BOLD_ENABLED:
        color = f"\033[1m{color}"
    if not color:
        return text
    return f"{color}{text}{ANSI_RESET}"


def format_user_text(text: str) -> str:
    """Apply optional ANSI style to transcribed user text."""
    if not supports_ansi():
        return text

    color = USER_COLOR_CODES.get(USER_TEXT_COLOR_NAME, "")
    if USER_TEXT_BOLD_ENABLED:
        color = f"\033[1m{color}"
    if not color:
        return text
    return f"{color}{text}{ANSI_RESET}"


def format_user_label(text: str) -> str:
    """Apply fixed ANSI style to the user speaker label."""
    if not supports_ansi():
        return text
    return f"\033[1m{USER_LABEL_COLOR}{text}{ANSI_RESET}"


def format_assistant_label(text: str) -> str:
    """Apply fixed ANSI style to the assistant speaker label."""
    if not supports_ansi():
        return text
    return f"\033[1m{ASSISTANT_LABEL_COLOR}{text}{ANSI_RESET}"


def record_wav_until_enter(path: Path, sample_rate: int, channels: int) -> None:
    """Record microphone audio until Enter is pressed, then save a WAV file."""
    chunks: list[np.ndarray] = []
    stop_event = threading.Event()

    def callback(
        indata: np.ndarray,
        _frames: int,
        _time: object,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            stderr(f"{status}\n")
        chunks.append(indata.copy())

    def wait_for_enter() -> None:
        with contextlib.suppress(EOFError):
            input()
        stop_event.set()

    stderr("Recording... press Enter to stop this turn.\n")
    input_thread = threading.Thread(target=wait_for_enter, daemon=True)
    input_thread.start()

    with sd.InputStream(
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
        callback=callback,
    ):
        while not stop_event.is_set():
            sd.sleep(100)

    if not chunks:
        msg = "No audio captured from microphone"
        raise RuntimeError(msg)

    audio = np.concatenate(chunks, axis=0)
    audio_int16 = np.clip(audio, -1, 1)
    audio_int16 = (audio_int16 * 32767).astype(np.int16)
    wav_write(path, sample_rate, audio_int16)


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


def capture_turn(args: argparse.Namespace) -> tuple[str, str | None]:
    """Record one microphone turn and transcribe it with Whisper."""

    def safe_session_dir_name(raw_name: str) -> str:
        """Convert session name to a filesystem-safe directory name."""
        cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", raw_name)
        return cleaned or "unknown-session"

    if args.keep_input_audio:
        session_name = getattr(args, "_input_audio_session", "new-session")
        session_dir = KEPT_INPUT_AUDIO_DIR / safe_session_dir_name(session_name)
        session_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        fd, temp_name = tempfile.mkstemp(
            prefix=f"{timestamp}-",
            suffix=".wav",
            dir=session_dir,
        )
        os.close(fd)
        wav_path = Path(temp_name)
        record_wav_until_enter(
            wav_path,
            sample_rate=args.input_sample_rate,
            channels=args.input_channels,
        )
        stderr("Transcribing...\n")
        text, detected_language = whisper_to_text(wav_path=wav_path, args=args)
        stderr(f"Saved recording: {wav_path}\n")
        return text.strip(), detected_language

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        wav_path = Path(tmp.name)
        record_wav_until_enter(
            wav_path,
            sample_rate=args.input_sample_rate,
            channels=args.input_channels,
        )
        stderr("Transcribing...\n")
        text, detected_language = whisper_to_text(wav_path=wav_path, args=args)
    return text.strip(), detected_language


# pylint: disable=too-many-branches,too-many-statements
def run_voice_chat(args: argparse.Namespace) -> None:  # noqa: C901, PLR0912, PLR0915
    """Run the continuous record/transcribe/ask/reply loop."""
    state_path = args.session_file.expanduser().resolve()
    session_id = resolve_session_id(args, state_path)
    speaker: KokoroSpeaker | None = None
    tts_enabled = args.voice

    if tts_enabled:
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
            args._input_audio_session = session_id or "new-session"
            user_text, detected_language = capture_turn(args)
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
            except (RuntimeError, ValueError, OSError, sd.PortAudioError) as exc:
                stderr(f"Kokoro playback failed: {exc}\n")


def main() -> None:
    """Entry point for voice chat."""
    args = parse_args()
    run_voice_chat(args)


if __name__ == "__main__":
    main()
