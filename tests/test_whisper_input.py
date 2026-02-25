"""Unit tests for Whisper input helper behavior."""

from __future__ import annotations

# pylint: disable=import-error  # E0401: some lint envs miss editable imports.
import argparse
import contextlib
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

from vincent import whisper_input

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    import pytest


def test_whisper_to_text_joins_nonempty_segments() -> None:
    """Join non-empty segment text values and return detected language."""

    class FakeWhisperModel:
        def transcribe(
            self, _wav_path: str, **_kwargs: object
        ) -> tuple[object, object]:
            segments = [
                SimpleNamespace(text=" hello "),
                SimpleNamespace(text=""),
                SimpleNamespace(text="world"),
            ]
            info = SimpleNamespace(language="en")
            return segments, info

    args = argparse.Namespace(whisper_task="transcribe", input_language=None)
    text, language = whisper_input.whisper_to_text(
        wav_path=Path("/tmp/fake.wav"),
        args=args,
        whisper_model=cast("Any", FakeWhisperModel()),
    )

    assert text == "hello world"
    assert language == "en"


def test_capture_turn_reports_transcribe_and_saved_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Emit status lines and return transcription from mocked helpers."""
    fake_path = tmp_path / "turn.wav"

    @contextlib.contextmanager
    def fake_turn_wav_path(_keep: bool, _session: str) -> Iterator[Path]:
        yield fake_path

    def fake_record_wav_until_enter(
        path: Path,
        sample_rate: int,
        channels: int,
        status_writer: Callable[[str], None],
    ) -> None:
        assert path == fake_path
        assert sample_rate == 16000
        assert channels == 1
        status_writer("Recording...\n")

    def fake_whisper_to_text(
        wav_path: Path,
        args: argparse.Namespace,
        whisper_model: object,
    ) -> tuple[str, str]:
        assert wav_path == fake_path
        assert args.keep_input_audio
        assert whisper_model is fake_model
        return "hello", "en"

    fake_model: Any = object()
    messages: list[str] = []
    args = argparse.Namespace(
        keep_input_audio=True,
        input_sample_rate=16000,
        input_channels=1,
    )

    monkeypatch.setattr(whisper_input, "turn_wav_path", fake_turn_wav_path)
    monkeypatch.setattr(
        whisper_input,
        "record_wav_until_enter",
        fake_record_wav_until_enter,
    )
    monkeypatch.setattr(whisper_input, "whisper_to_text", fake_whisper_to_text)

    text, language = whisper_input.capture_turn(
        args=args,
        input_audio_session="ses_123",
        whisper_model=fake_model,
        status_writer=messages.append,
    )

    assert text == "hello"
    assert language == "en"
    assert any("Transcribing" in message for message in messages)
    assert any("Saved recording" in message for message in messages)
