"""Small CLI helpers for inspecting Kokoro voices and language codes."""

from __future__ import annotations

import argparse


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line options for Kokoro metadata inspection."""
    parser = argparse.ArgumentParser(
        description="List Kokoro voices and language codes.",
    )
    parser.add_argument(
        "--repo-id",
        default="hexgrad/Kokoro-82M",
        help="Hugging Face repo id that contains voice files",
    )
    parser.add_argument(
        "--voices",
        action="store_true",
        help="List available Kokoro voices from the repo",
    )
    parser.add_argument(
        "--lang-codes",
        action="store_true",
        help="List Kokoro language codes and names",
    )
    parser.add_argument(
        "--aliases",
        action="store_true",
        help="List Kokoro language aliases",
    )
    return parser.parse_args(argv)


def list_voices(repo_id: str) -> list[str]:
    """Return available voice ids from a Kokoro Hugging Face repository."""
    from huggingface_hub import list_repo_files

    files = list_repo_files(repo_id=repo_id, repo_type="model")
    voices = [
        path.removeprefix("voices/").removesuffix(".pt")
        for path in files
        if path.startswith("voices/") and path.endswith(".pt")
    ]
    voices.sort()
    return voices


def list_lang_codes() -> dict[str, str]:
    """Return Kokoro language code mapping."""
    from kokoro.pipeline import LANG_CODES

    return dict(LANG_CODES)


def list_aliases() -> dict[str, str]:
    """Return Kokoro language alias mapping."""
    from kokoro.pipeline import ALIASES

    return dict(ALIASES)


def main(argv: list[str] | None = None) -> int:
    """Run metadata commands and print requested results."""
    args = parse_args(argv)
    show_any = args.voices or args.lang_codes or args.aliases

    if not show_any or args.lang_codes:
        lang_codes = list_lang_codes()
        print("Language Codes:")
        for code, name in sorted(lang_codes.items()):
            print(f"- {code}: {name}")
        print()

    if not show_any or args.aliases:
        aliases = list_aliases()
        print("Language Aliases:")
        for alias, code in sorted(aliases.items()):
            print(f"- {alias} -> {code}")
        print()

    if not show_any or args.voices:
        try:
            voices = list_voices(args.repo_id)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"Could not list voices from {args.repo_id}: {exc}")
            return 1
        print(f"Voices ({len(voices)}):")
        for voice in voices:
            print(f"- {voice}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
