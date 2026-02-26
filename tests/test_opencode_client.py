"""Unit tests for opencode command and event parsing helpers."""

from __future__ import annotations

# pylint: disable=import-error  # E0401: some lint envs miss editable imports.
import subprocess

import pytest

from vincent import opencode_client


def test_parse_opencode_events_collects_text_and_session() -> None:
    """Combine text chunks and return the latest discovered session id."""
    output = (
        '{"type":"status","sessionID":"ses_old"}\n'
        '{"type":"text","sessionID":"ses_new","part":{"text":"Hello "}}\n'
        '{"type":"text","part":{"text":"world"}}\n'
        '{"type":"tool","part":{"text":"ignored"}}\n'
        "this is not json"
    )

    response_text, session_id = opencode_client.parse_opencode_events(output)

    assert response_text == "Hello world"
    assert session_id == "ses_new"


def test_build_opencode_command_includes_optional_flags() -> None:
    """Build command arguments with session and optional opencode settings."""
    command = opencode_client.build_opencode_command(
        message="hello",
        options=opencode_client.OpenCodeRunOptions(
            session_id="ses_123",
            model="provider/model",
            agent="helper",
            attach="http://127.0.0.1:4096",
            directory="workdir",
        ),
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


def test_ask_opencode_reports_missing_executable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise a clear RuntimeError when `opencode` is unavailable."""

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError

    monkeypatch.setattr(opencode_client.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="`opencode` executable was not found"):
        opencode_client.ask_opencode(
            "hello",
            options=opencode_client.OpenCodeRunOptions(
                session_id=None,
                model=None,
                agent=None,
                attach=None,
                directory=None,
            ),
        )


def test_ask_opencode_reports_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raise RuntimeError containing stderr/stdout details on command failure."""
    failed = subprocess.CompletedProcess(
        args=["opencode", "run"],
        returncode=7,
        stdout="",
        stderr="boom",
    )

    monkeypatch.setattr(
        opencode_client.subprocess,
        "run",
        lambda *_args, **_kwargs: failed,
    )

    with pytest.raises(RuntimeError, match=r"opencode run failed \(7\): boom"):
        opencode_client.ask_opencode(
            "hello",
            options=opencode_client.OpenCodeRunOptions(
                session_id="ses_1",
                model=None,
                agent=None,
                attach=None,
                directory=None,
            ),
        )
