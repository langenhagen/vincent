"""Microphone recording and input-file path utilities.

Handles low-level audio-input concerns.
"""

from __future__ import annotations

import contextlib
import os
import re
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wav_write

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

KEPT_INPUT_AUDIO_DIR = Path(".voice_inputs")


def safe_session_dir_name(raw_name: str) -> str:
    """Convert a session id/name to a filesystem-safe directory name."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", raw_name)
    return cleaned or "unknown-session"


def create_kept_input_path(input_audio_session: str) -> Path:
    """Create a stable, timestamped WAV path for kept input audio."""
    session_dir = KEPT_INPUT_AUDIO_DIR / safe_session_dir_name(input_audio_session)
    session_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=UTC).strftime("%Y-%m-%d-%H-%M-%S")
    fd, temp_name = tempfile.mkstemp(
        prefix=f"{timestamp}-",
        suffix=".wav",
        dir=session_dir,
    )
    os.close(fd)
    return Path(temp_name)


@contextlib.contextmanager
def turn_wav_path(
    keep_input_audio: bool,
    input_audio_session: str,
) -> Iterator[Path]:
    """Yield a temporary or persisted WAV file path for one turn."""
    if keep_input_audio:
        yield create_kept_input_path(input_audio_session)
        return

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        yield Path(tmp.name)


def record_wav_until_enter(
    path: Path,
    sample_rate: int,
    channels: int,
    status_writer: Callable[[str], None],
) -> None:
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
            status_writer(f"{status}\n")
        chunks.append(indata.copy())

    def wait_for_enter() -> None:
        with contextlib.suppress(EOFError):
            input()
        stop_event.set()

    status_writer("Recording... press Enter to stop this turn.\n")
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
