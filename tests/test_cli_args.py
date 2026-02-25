"""CLI argument parsing tests for the main Vincent command."""

from __future__ import annotations

# pylint: disable=import-error  # E0401: some lint envs miss editable imports.
import sys

import pytest

from vincent import cli

EXPECTED_SAMPLE_RATE = 16_000


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

    args = cli.parse_args()

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
        cli.parse_args()
    except SystemExit:
        return

    msg = "Expected parse_args to exit for removed --model flag"
    raise AssertionError(msg)


def test_parse_args_voice_defaults_on_and_can_disable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enable voice by default and allow explicit disable via --no-voice."""
    monkeypatch.setattr(sys, "argv", ["vincent"])
    default_args = cli.parse_args()

    monkeypatch.setattr(sys, "argv", ["vincent", "--no-voice"])
    disabled_args = cli.parse_args()

    assert default_args.voice
    assert not disabled_args.voice


def test_parse_args_rejects_non_positive_input_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject zero/negative values for audio input numeric options."""
    monkeypatch.setattr(sys, "argv", ["vincent", "--input-sample-rate", "0"])
    with pytest.raises(SystemExit):
        cli.parse_args()

    monkeypatch.setattr(sys, "argv", ["vincent", "--input-channels", "-1"])
    with pytest.raises(SystemExit):
        cli.parse_args()
