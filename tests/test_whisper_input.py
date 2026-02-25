"""Unit tests for Whisper input helper behavior."""

from __future__ import annotations

# pylint: disable=import-error  # E0401: some lint envs miss editable imports.
import argparse
import contextlib
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

from vincent import whisper_input

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

EXPECTED_SAMPLE_RATE = 16_000


def test_whisper_to_text_joins_nonempty_segments(tmp_path: Path) -> None:
    """Join non-empty segment text values and return detected language."""

    class FakeWhisperModel:  # pylint: disable=too-few-public-methods
        """Simple stand-in that exposes the Whisper transcribe interface."""

        def transcribe(
            self,
            _wav_path: str,
            **_kwargs: object,
        ) -> tuple[object, object]:
            """Return fixed segment and language values for testing."""
            segments = [
                SimpleNamespace(text=" hello "),
                SimpleNamespace(text=""),
                SimpleNamespace(text="world"),
            ]
            info = SimpleNamespace(language="en")
            return segments, info

    args = argparse.Namespace(whisper_task="transcribe", input_language=None)
    text, language = whisper_input.whisper_to_text(
        wav_path=tmp_path / "fake.wav",
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
    def fake_turn_wav_path(
        *,
        keep_input_audio: bool,
        input_audio_session: str,
    ) -> Iterator[Path]:
        assert keep_input_audio
        assert input_audio_session == "ses_123"
        yield fake_path

    def fake_record_wav_until_enter(
        path: Path,
        sample_rate: int,
        channels: int,
        status_writer: Callable[[str], None],
    ) -> None:
        assert path == fake_path
        assert sample_rate == EXPECTED_SAMPLE_RATE
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
        whisper_model=cast("Any", fake_model),
        status_writer=messages.append,
    )

    assert text == "hello"
    assert language == "en"
    assert any("Transcribing" in message for message in messages)
    assert any("Saved recording" in message for message in messages)


def test_capture_turn_removes_kept_file_on_record_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Delete retained path when recording fails before transcription."""
    wav_path = tmp_path / "turn.wav"
    wav_path.write_text("placeholder", encoding="utf-8")

    @contextlib.contextmanager
    def fake_turn_wav_path(
        *,
        keep_input_audio: bool,
        input_audio_session: str,
    ) -> Iterator[Path]:
        assert keep_input_audio
        assert input_audio_session == "ses_123"
        yield wav_path

    def fake_record_wav_until_enter(
        path: Path,
        sample_rate: int,
        channels: int,
        status_writer: Callable[[str], None],
    ) -> None:
        assert path == wav_path
        assert sample_rate == EXPECTED_SAMPLE_RATE
        assert channels == 1
        status_writer("Recording...\n")
        msg = "No audio captured from microphone"
        raise RuntimeError(msg)

    args = argparse.Namespace(
        keep_input_audio=True,
        input_sample_rate=EXPECTED_SAMPLE_RATE,
        input_channels=1,
    )

    monkeypatch.setattr(whisper_input, "turn_wav_path", fake_turn_wav_path)
    monkeypatch.setattr(
        whisper_input,
        "record_wav_until_enter",
        fake_record_wav_until_enter,
    )

    with pytest.raises(RuntimeError):
        whisper_input.capture_turn(
            args=args,
            input_audio_session="ses_123",
            whisper_model=cast("Any", object()),
            status_writer=lambda _msg: None,
        )

    assert not wav_path.exists()
