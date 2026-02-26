"""Thin opencode CLI client helpers.

This module isolates command construction, JSON event parsing, and process
execution for `opencode run` so the main CLI loop can stay focused on chat
orchestration.
"""

from __future__ import annotations

import json
import subprocess  # nosec B404  # B404: required for opencode CLI subprocess call.
from dataclasses import dataclass


@dataclass(frozen=True)
class OpenCodeRunOptions:
    """Options used to build and run one opencode request."""

    session_id: str | None
    model: str | None
    agent: str | None
    attach: str | None
    directory: str | None


def build_opencode_command(
    message: str,
    options: OpenCodeRunOptions,
) -> list[str]:
    """Build the opencode command argv for one conversation turn."""
    command = ["opencode", "run", "--format", "json"]
    if options.session_id:
        command.extend(["--session", options.session_id])
    if options.model:
        command.extend(["--model", options.model])
    if options.agent:
        command.extend(["--agent", options.agent])
    if options.attach:
        command.extend(["--attach", options.attach])
    if options.directory:
        command.extend(["--dir", options.directory])
    command.append(message)
    return command


def parse_opencode_events(output: str) -> tuple[str, str | None]:
    """Parse JSON event lines and return response text plus session id."""
    response_chunks: list[str] = []
    discovered_session: str | None = None

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        event_session_id = event.get("sessionID")
        if isinstance(event_session_id, str) and event_session_id:
            discovered_session = event_session_id

        if event.get("type") != "text":
            continue

        part = event.get("part")
        if not isinstance(part, dict):
            continue

        text = part.get("text")
        if isinstance(text, str) and text:
            response_chunks.append(text)

    return "".join(response_chunks).strip(), discovered_session


def ask_opencode(
    prompt: str,
    options: OpenCodeRunOptions,
) -> tuple[str, str | None]:
    """Send one prompt to opencode and return response text and session id."""
    command = build_opencode_command(prompt, options)
    try:
        # Fixed argv list; shell execution is explicitly disabled.
        result = subprocess.run(  # noqa: S603  # nosec B603
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        msg = "`opencode` executable was not found in PATH"
        raise RuntimeError(msg) from exc
    except OSError as exc:
        msg = f"Failed to launch `opencode`: {exc}"
        raise RuntimeError(msg) from exc

    if result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip()
        msg = f"opencode run failed ({result.returncode}): {details}"
        raise RuntimeError(msg)

    response_text, discovered_session = parse_opencode_events(result.stdout)
    return response_text, discovered_session or options.session_id
