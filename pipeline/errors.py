"""Pipeline errors with actionable install hints."""

from __future__ import annotations


class PipelineError(Exception):
    pass


class MissingDependencyError(PipelineError):
    """Raised when a required external tool or Python package is missing.

    Attach an `install_hint` so the CLI can show the user how to fix it.
    """

    def __init__(self, what: str, install_hint: str) -> None:
        super().__init__(f"Missing dependency: {what}\n  {install_hint}")
        self.what = what
        self.install_hint = install_hint


class DownloadError(PipelineError):
    """Raised when spotdl fails to match a song to a YouTube source."""


INSTALL_HINTS = {
    "ffmpeg": "Install ffmpeg: `brew install ffmpeg` (macOS) or `apt install ffmpeg` (Linux).",
    "spotdl": "Install spotdl: `uv sync` should install it. If it's missing, run `uv pip install spotdl`.",
    "demucs": "Install demucs: `uv sync` should install it. If missing, `uv pip install demucs`.",
    "basic_pitch": "Install basic-pitch: `uv sync`. If TF wheels fail on Apple Silicon, "
    "see https://github.com/spotify/basic-pitch for the tensorflow-macos workaround.",
    "ADTOF": (
        "Install ADTOF: `uv sync --extra adtof` (installs from "
        "https://github.com/MZehren/ADTOF). If that fails, see "
        "https://github.com/MZehren/ADTOF#installation for setup; "
        "or fall back to magenta with `uv sync --extra magenta`."
    ),
}
