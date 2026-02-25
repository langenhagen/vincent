"""Tests for Kokoro metadata helper CLI utilities."""

from __future__ import annotations

# pylint: disable=import-error  # E0401: some lint envs miss editable imports.
from typing import TYPE_CHECKING

from vincent import kokoro_info

if TYPE_CHECKING:
    import pytest


def test_list_voices_filters_and_sorts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return only voice files and sort resulting voice ids."""

    def fake_list_repo_files(*, repo_id: str, repo_type: str) -> list[str]:
        assert repo_id == "hexgrad/Kokoro-82M"
        assert repo_type == "model"
        return [
            "README.md",
            "voices/zf_xiaobei.pt",
            "voices/af_heart.pt",
            "voices/not-a-voice.txt",
        ]

    monkeypatch.setattr(
        kokoro_info,
        "list_repo_files",
        fake_list_repo_files,
    )

    voices = kokoro_info.list_voices("hexgrad/Kokoro-82M")

    assert voices == ["af_heart", "zf_xiaobei"]


def test_main_prints_requested_sections(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Print selected sections and return success status."""
    monkeypatch.setattr(
        kokoro_info,
        "list_lang_codes",
        lambda: {"a": "American English"},
    )
    monkeypatch.setattr(
        kokoro_info,
        "list_aliases",
        lambda: {"en-us": "a"},
    )
    monkeypatch.setattr(
        kokoro_info,
        "list_voices",
        lambda _repo_id: ["af_heart"],
    )

    exit_code = kokoro_info.main(["--lang-codes", "--aliases", "--voices"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Language Codes:" in output
    assert "- a: American English" in output
    assert "Language Aliases:" in output
    assert "- en-us -> a" in output
    assert "Voices (1):" in output
    assert "- af_heart" in output


def test_main_without_flags_prints_all_sections(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Show every section when no explicit output flags are provided."""
    monkeypatch.setattr(
        kokoro_info,
        "list_lang_codes",
        lambda: {"a": "American English"},
    )
    monkeypatch.setattr(
        kokoro_info,
        "list_aliases",
        lambda: {"en-us": "a"},
    )
    monkeypatch.setattr(
        kokoro_info,
        "list_voices",
        lambda _repo_id: ["af_heart"],
    )

    exit_code = kokoro_info.main([])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Language Codes:" in output
    assert "Language Aliases:" in output
    assert "Voices (1):" in output
