"""Integration-style CLI loop tests with heavy mocking.

These tests run the main loop orchestration while mocking device/network
boundaries so behavior can be validated without a microphone, speaker, or
opencode process.
"""

from __future__ import annotations

# pylint: disable=import-error  # E0401: some lint envs miss editable imports.
import argparse
from typing import TYPE_CHECKING

from vincent import cli

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

    from vincent.opencode_client import OpenCodeRunOptions


def make_args(tmp_path: Path) -> argparse.Namespace:
    """Create a complete argparse namespace for run_voice_chat tests."""
    return argparse.Namespace(
        whisper_model="base",
        whisper_device="cpu",
        whisper_compute_type="int8",
        whisper_task="transcribe",
        input_language=None,
        input_sample_rate=16000,
        input_channels=1,
        keep_input_audio=False,
        session_id=None,
        new_session=False,
        session_file=tmp_path / "state.json",
        opencode_model=None,
        opencode_agent=None,
        opencode_attach=None,
        opencode_dir=None,
        voice=False,
        tts_voice="af_heart",
        tts_lang_code="a",
        tts_speed=1.0,
    )


def test_run_voice_chat_success_path_persists_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Save discovered session id and print assistant output on success."""
    turns = iter([("hello", "en"), ("quit", "en")])
    output_messages: list[str] = []
    error_messages: list[str] = []

    def fake_ask_opencode(
        prompt: str,
        *,
        options: OpenCodeRunOptions,
    ) -> tuple[str, str | None]:
        assert prompt == "hello"
        assert options.session_id is None
        assert options.model is None
        assert options.agent is None
        assert options.attach is None
        assert options.directory is None
        return "Hello back", "ses_new"

    monkeypatch.setattr(cli, "build_whisper_model", lambda _args: object())
    monkeypatch.setattr(
        cli,
        "capture_turn",
        lambda _args, _session, _model, _status: next(turns),
    )
    monkeypatch.setattr(
        cli,
        "ask_opencode",
        fake_ask_opencode,
    )
    monkeypatch.setattr(cli, "stdout", output_messages.append)
    monkeypatch.setattr(cli, "stderr", error_messages.append)

    args = make_args(tmp_path)
    cli.run_voice_chat(args)

    assert cli.load_session_id(args.session_file) == "ses_new"
    assert any("Assistant:" in message for message in output_messages)
    assert any("Hello back" in message for message in output_messages)
    assert any(
        "Saved opencode session: ses_new" in message for message in error_messages
    )


def test_run_voice_chat_opencode_error_logs_and_continues(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Log opencode errors and continue to next user turn without crashing."""
    turns = iter([("hello", "en"), ("quit", "en")])
    output_messages: list[str] = []
    error_messages: list[str] = []

    monkeypatch.setattr(cli, "build_whisper_model", lambda _args: object())
    monkeypatch.setattr(
        cli,
        "capture_turn",
        lambda _args, _session, _model, _status: next(turns),
    )

    def fake_ask_opencode(
        prompt: str,
        *,
        options: OpenCodeRunOptions,
    ) -> tuple[str, str | None]:
        assert prompt == "hello"
        assert options.session_id is None
        assert options.model is None
        assert options.agent is None
        assert options.attach is None
        assert options.directory is None
        msg = "opencode boom"
        raise RuntimeError(msg)

    monkeypatch.setattr(cli, "ask_opencode", fake_ask_opencode)
    monkeypatch.setattr(cli, "stdout", output_messages.append)
    monkeypatch.setattr(cli, "stderr", error_messages.append)

    args = make_args(tmp_path)
    cli.run_voice_chat(args)

    assert not args.session_file.exists()
    assert any("opencode boom" in message for message in error_messages)
    assert not any("Hello back" in message for message in output_messages)
