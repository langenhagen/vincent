"""Unit tests for opencode command and event parsing helpers."""

from __future__ import annotations

# pylint: disable=import-error  # E0401: some lint envs miss editable imports.
from vincent import cli


def test_parse_opencode_events_collects_text_and_session() -> None:
    """Combine text chunks and return the latest discovered session id."""
    output = (
        '{"type":"status","sessionID":"ses_old"}\n'
        '{"type":"text","sessionID":"ses_new","part":{"text":"Hello "}}\n'
        '{"type":"text","part":{"text":"world"}}\n'
        '{"type":"tool","part":{"text":"ignored"}}\n'
        "this is not json"
    )

    response_text, session_id = cli.parse_opencode_events(output)

    assert response_text == "Hello world"
    assert session_id == "ses_new"


def test_build_opencode_command_includes_optional_flags() -> None:
    """Build command arguments with session and optional opencode settings."""
    args = cli.argparse.Namespace(
        opencode_model="provider/model",
        opencode_agent="helper",
        opencode_attach="http://127.0.0.1:4096",
        opencode_dir="workdir",
    )

    command = cli.build_opencode_command(
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
