"""Small CLI helpers for inspecting Kokoro voices and language codes."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from typing import Literal, overload


def stdout(message: str) -> None:
    """Write text to stdout without adding extra formatting."""
    sys.stdout.write(message)


@overload
def load_module_attr(
    module_name: Literal["huggingface_hub"],
    attr_name: Literal["list_repo_files"],
) -> Callable[..., list[str]]: ...


@overload
def load_module_attr(
    module_name: Literal["kokoro.pipeline"],
    attr_name: Literal["LANG_CODES"],
) -> dict[str, str]: ...


@overload
def load_module_attr(
    module_name: Literal["kokoro.pipeline"],
    attr_name: Literal["ALIASES"],
) -> dict[str, str]: ...


def load_module_attr(module_name: str, attr_name: str) -> object:
    """Load a whitelisted attribute from known modules."""
    if module_name == "huggingface_hub" and attr_name == "list_repo_files":
        from huggingface_hub import list_repo_files

        return list_repo_files

    if module_name == "kokoro.pipeline" and attr_name == "LANG_CODES":
        from kokoro.pipeline import LANG_CODES

        return LANG_CODES

    if module_name == "kokoro.pipeline" and attr_name == "ALIASES":
        from kokoro.pipeline import ALIASES

        return ALIASES

    msg = f"Unsupported module attribute: {module_name}.{attr_name}"
    raise ValueError(msg)


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
    list_repo_files = load_module_attr("huggingface_hub", "list_repo_files")
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
    return load_module_attr("kokoro.pipeline", "LANG_CODES").copy()


def list_aliases() -> dict[str, str]:
    """Return Kokoro language alias mapping."""
    return load_module_attr("kokoro.pipeline", "ALIASES").copy()


def main(argv: list[str] | None = None) -> int:
    """Run metadata commands and print requested results."""
    args = parse_args(argv)
    show_any = args.voices or args.lang_codes or args.aliases

    if not show_any or args.lang_codes:
        lang_codes = list_lang_codes()
        stdout("Language Codes:\n")
        for code, name in sorted(lang_codes.items()):
            stdout(f"- {code}: {name}\n")
        stdout("\n")

    if not show_any or args.aliases:
        aliases = list_aliases()
        stdout("Language Aliases:\n")
        for alias, code in sorted(aliases.items()):
            stdout(f"- {alias} -> {code}\n")
        stdout("\n")

    if not show_any or args.voices:
        voices = list_voices(args.repo_id)
        stdout(f"Voices ({len(voices)}):\n")
        for voice in voices:
            stdout(f"- {voice}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
