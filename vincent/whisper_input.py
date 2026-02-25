"""Whisper-side input processing helpers.

Owns Whisper model lifecycle and per-turn transcription flow. It bridges
recorded WAV files into recognized text.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from faster_whisper import WhisperModel

from .audio_recording import record_wav_until_enter, turn_wav_path

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable
    from pathlib import Path


def build_whisper_model(args: argparse.Namespace) -> WhisperModel:
    """Build one Whisper model instance reused across turns."""
    return WhisperModel(
        args.whisper_model,
        device=args.whisper_device,
        compute_type=args.whisper_compute_type,
    )


def whisper_to_text(
    wav_path: Path,
    args: argparse.Namespace,
    whisper_model: WhisperModel,
) -> tuple[str, str | None]:
    """Run Whisper on a WAV file and return text plus detected language."""
    segments, info = whisper_model.transcribe(
        str(wav_path),
        task=args.whisper_task,
        language=args.input_language,
        vad_filter=True,
    )
    text = " ".join(
        segment.text.strip() for segment in segments if segment.text.strip()
    )
    return text.strip(), getattr(info, "language", None)


def capture_turn(
    args: argparse.Namespace,
    input_audio_session: str,
    whisper_model: WhisperModel,
    status_writer: Callable[[str], None],
) -> tuple[str, str | None]:
    """Record one turn from the mic and transcribe it with Whisper."""
    with turn_wav_path(
        keep_input_audio=args.keep_input_audio,
        input_audio_session=input_audio_session,
    ) as wav_path:
        try:
            record_wav_until_enter(
                wav_path,
                sample_rate=args.input_sample_rate,
                channels=args.input_channels,
                status_writer=status_writer,
            )
            status_writer("Transcribing...\n")
            text, detected_language = whisper_to_text(
                wav_path=wav_path,
                args=args,
                whisper_model=whisper_model,
            )
        except Exception:
            if args.keep_input_audio:
                with contextlib.suppress(OSError):
                    wav_path.unlink()
            raise

        if args.keep_input_audio:
            status_writer(f"Saved recording: {wav_path}\n")

    return text.strip(), detected_language
