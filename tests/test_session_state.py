"""Unit tests for opencode session state persistence helpers."""

from __future__ import annotations

# pylint: disable=import-error  # E0401: some lint envs miss editable imports.
import argparse
from typing import TYPE_CHECKING

from vincent import cli

if TYPE_CHECKING:
    from pathlib import Path


def test_save_and_load_session_id_roundtrip(tmp_path: Path) -> None:
    """Persist and reload a valid session id from disk."""
    state_path = tmp_path / "state.json"
    cli.save_session_id(state_path, "ses_123")

    loaded = cli.load_session_id(state_path)

    assert loaded == "ses_123"


def test_load_session_id_handles_missing_and_invalid_files(tmp_path: Path) -> None:
    """Return None for missing file and invalid JSON content."""
    missing = tmp_path / "missing.json"
    assert cli.load_session_id(missing) is None

    broken = tmp_path / "broken.json"
    broken.write_text("{not-valid-json", encoding="utf-8")
    assert cli.load_session_id(broken) is None


def test_resolve_session_id_prefers_explicit_session(tmp_path: Path) -> None:
    """Use explicit --session-id and persist it to the state file."""
    state_path = tmp_path / "state.json"
    args = argparse.Namespace(session_id="ses_explicit", new_session=False)

    resolved = cli.resolve_session_id(args, state_path)

    assert resolved == "ses_explicit"
    assert cli.load_session_id(state_path) == "ses_explicit"


def test_resolve_session_id_honors_new_session_flag(tmp_path: Path) -> None:
    """Return no session id when --new-session is requested."""
    state_path = tmp_path / "state.json"
    cli.save_session_id(state_path, "ses_existing")
    args = argparse.Namespace(session_id=None, new_session=True)

    resolved = cli.resolve_session_id(args, state_path)

    assert resolved is None


def test_resolve_session_id_uses_saved_value_by_default(tmp_path: Path) -> None:
    """Reuse saved session id when no overriding flags are provided."""
    state_path = tmp_path / "state.json"
    cli.save_session_id(state_path, "ses_saved")
    args = argparse.Namespace(session_id=None, new_session=False)

    resolved = cli.resolve_session_id(args, state_path)

    assert resolved == "ses_saved"
