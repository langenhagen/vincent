# AGENTS.md

Practical guidance for humans and coding agents working in this repository.

## Intent

- Keep changes small, reviewable, and aligned with existing repo conventions.
- Prefer reliable, non-interactive commands and deterministic output.
- Assume files may change while you work; re-read before final writes and commits.
- Search the web proactively when debugging, investigating, or verifying API behavior.
- Add clear names and comments where structure is non-obvious.
- Avoid over-engineering: prefer the simplest solution that is clear, maintainable, and testable.
- Capitalize Markdown headlines (prefer Title Case).

## Project Stack

- Language: Python.
- Runtime/version management: `pyenv`.
- Package/dependency/task workflow: `uv`.
- Core features: microphone capture, Whisper transcription, opencode session bridge, optional Kokoro TTS.
- Test framework: `pytest`.
- Lint/format: `ruff`.
- Type checking: `mypy`.

## Repo Layout

- `voice_chat.py`: main CLI entrypoint.
- `README.md`: usage and setup notes.
- `pyproject.toml`: metadata and tooling configuration.
- `.python-version`: pinned local Python version.

## Local Workflow

Prefer repo-local, reproducible commands:

- Sync dependencies: `uv sync`.
- Run app: `uv run voice-chat` (or `uv run python voice_chat.py`).
- Run app with assistant speech output: `uv run voice-chat --voice`.
- Run tests: `uv run pytest`.
- Run linter: `uv run ruff check .`.
- Format code: `uv run ruff format .`.
- Run type checks: `uv run mypy`.
- Install hooks: `uv run pre-commit install`.
- Run hooks manually: `uv run pre-commit run --all-files`.
- Optional extended lint checks: `uv run --group lint pylint voice_chat.py`.
- Optional personal lint sweep: `source .venv/bin/activate && l3`.
- Optional personal autofix pass: `source .venv/bin/activate && rf`.

Do not run full test suites automatically unless requested; use focused checks for touched files first when possible.

## Testing Boundaries

- Prefer unit tests for parsing, command building, and pure logic.
- Do not write tests that require a live microphone, speaker, or audio device by default.
- Do not write tests that call `opencode` or external APIs/networks by default.
- Mock external process, network, and device dependencies in tests.
- If an integration test against `opencode` is truly needed, document cost/latency tradeoffs and run it sparingly.

## User Shorthand Conventions

Interpret these tokens as explicit workflow commands:

- `prose`
  - Provide a clear prose walkthrough of the topic or changes.
  - Prioritize rationale, tradeoffs, and how pieces fit together.

- `eli5` or `eli`
  - Provide a short, simple, technically correct explanation.

- `sw`
  - Explicitly search the web before answering.
  - Use web results as supporting context.

- `mc`, `cm`, or `commit`
  - Create at least one git commit.
  - Follow commit message rules below.
  - Include both the commit message and a short prose walkthrough.

## Commit Workflow Expectations

When asked to commit:

1. Inspect `git status`, full diff, and recent commit style.
2. Stage only relevant files.
3. Run focused checks for touched areas.
4. Commit with a plain imperative summary line, no Conventional Commit prefix.
5. Report commit hash, message, and a short prose walkthrough.

Commit message DO:

- Start the summary in imperative mood (for example `Add`, `Fix`, `Change`).
- Keep the second line blank.
- Wrap body text to about 72 columns.
- Use real line breaks in the body; do not pass wrapped text as one line.
- When using shell commits, prefer multiline `-m $'line1\nline2'` or a heredoc so wraps are preserved.

Commit message DON'T:

- Do not use Conventional Commit prefixes.
- Do not end the summary line with a period.
- Do not include literal `\n` text in commit messages.

Before finalizing a commit, verify formatting with:

- `git log -1 --pretty=%B`
- Ensure no body line exceeds roughly 72 characters.

Never include secrets in commits (`.env*`, tokens, private keys, auth dumps).

## Git and Editing Safety

- Do not revert unrelated user changes.
- Do not use destructive git commands unless explicitly requested.
- Avoid interactive commands in automation.
- If a patch fails or context looks stale, re-read files before retrying.

## Linter Pragmas

- Keep suppressions (`noqa`, `nosec`, `type: ignore`, pylint disables) as narrow as possible.
- Document suppressions inline with both the rule meaning and short local reason.
- Prefer config-level ignores for broad patterns over repeated inline suppressions.

## Output Character Policy

- Prefer plain ASCII in output and docs unless a file already requires Unicode.
- Avoid fancy punctuation and hidden/special spacing characters.
- Normalize pasted external text to plain characters before finalizing.
