"""Unit tests for non-device audio path helpers."""

from __future__ import annotations

# pylint: disable=import-error  # E0401: some lint envs miss editable imports.
from pathlib import Path

from vincent import audio_recording


def test_safe_session_dir_name_sanitizes_special_chars() -> None:
    """Replace path-unsafe characters with underscores."""
    cleaned = audio_recording.safe_session_dir_name("ses:/with*odd?chars")
    assert cleaned == "ses__with_odd_chars"


def test_turn_wav_path_uses_temp_file_when_not_kept() -> None:
    """Yield a temporary .wav path for one non-persisted turn."""
    with audio_recording.turn_wav_path(
        keep_input_audio=False,
        input_audio_session="ses_123",
    ) as wav_path:
        assert isinstance(wav_path, Path)
        assert wav_path.suffix == ".wav"


def test_turn_wav_path_persists_under_session_folder(tmp_path: Path) -> None:
    """Place kept wav files under .voice_inputs/<session>."""
    original_dir = audio_recording.KEPT_INPUT_AUDIO_DIR
    audio_recording.KEPT_INPUT_AUDIO_DIR = tmp_path
    try:
        with audio_recording.turn_wav_path(
            keep_input_audio=True,
            input_audio_session="ses_abc",
        ) as wav_path:
            assert wav_path.parent == tmp_path / "ses_abc"
            assert wav_path.name.endswith(".wav")
    finally:
        audio_recording.KEPT_INPUT_AUDIO_DIR = original_dir
