"""Core behavior tests for the voice chat CLI module."""

from __future__ import annotations

# pylint: disable=import-error  # E0401: some lint envs miss editable imports.
import sys
from typing import TYPE_CHECKING

import voice_chat

if TYPE_CHECKING:
    import pytest

EXPECTED_SAMPLE_RATE = 16_000


def test_parse_opencode_events_collects_text_and_session() -> None:
    """Combine text chunks and return the latest discovered session id."""
    output = (
        '{"type":"status","sessionID":"ses_old"}\n'
        '{"type":"text","sessionID":"ses_new","part":{"text":"Hello "}}\n'
        '{"type":"text","part":{"text":"world"}}\n'
        '{"type":"tool","part":{"text":"ignored"}}\n'
        "this is not json"
    )

    response_text, session_id = voice_chat.parse_opencode_events(output)

    assert response_text == "Hello world"
    assert session_id == "ses_new"


def test_build_opencode_command_includes_optional_flags() -> None:
    """Build command arguments with session and optional opencode settings."""
    args = voice_chat.argparse.Namespace(
        opencode_model="provider/model",
        opencode_agent="helper",
        opencode_attach="http://127.0.0.1:4096",
        opencode_dir="workdir",
    )

    command = voice_chat.build_opencode_command(
        message="hello",
        args=args,
        session_id="ses_123",
    )

    assert command == [
        "opencode",
        "run",
        "--format",
        "json",
        "--session",
        "ses_123",
        "--model",
        "provider/model",
        "--agent",
        "helper",
        "--attach",
        "http://127.0.0.1:4096",
        "--dir",
        "workdir",
        "hello",
    ]


def test_parse_args_accepts_renamed_input_and_whisper_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accept renamed input and Whisper options and map argparse fields."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vincent",
            "--whisper-model",
            "small",
            "--whisper-device",
            "cpu",
            "--whisper-compute-type",
            "float32",
            "--whisper-task",
            "translate",
            "--input-language",
            "en",
            "--input-sample-rate",
            "16000",
            "--input-channels",
            "1",
            "--keep-input-audio",
        ],
    )

    args = voice_chat.parse_args()

    assert args.whisper_model == "small"
    assert args.whisper_device == "cpu"
    assert args.whisper_compute_type == "float32"
    assert args.whisper_task == "translate"
    assert args.input_language == "en"
    assert args.input_sample_rate == EXPECTED_SAMPLE_RATE
    assert args.input_channels == 1
    assert args.keep_input_audio


def test_parse_args_rejects_old_model_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject deprecated short Whisper options removed from the CLI."""
    monkeypatch.setattr(sys, "argv", ["vincent", "--model", "small"])

    try:
        voice_chat.parse_args()
    except SystemExit:
        return

    msg = "Expected parse_args to exit for removed --model flag"
    raise AssertionError(msg)


def test_parse_args_voice_defaults_on_and_can_disable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enable voice by default and allow explicit disable via --no-voice."""
    monkeypatch.setattr(sys, "argv", ["vincent"])
    default_args = voice_chat.parse_args()

    monkeypatch.setattr(sys, "argv", ["vincent", "--no-voice"])
    disabled_args = voice_chat.parse_args()

    assert default_args.voice
    assert not disabled_args.voice
