"""Command-line interface.

Single entry point:
    python -m pipeline <spotify_playlist_url> --output ./output
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from pipeline.errors import MissingDependencyError
from pipeline.runner import ALL_STEPS, run_pipeline
from pipeline.storage import LocalStorage

app = typer.Typer(
    add_completion=False,
    help="Spotify playlist -> stems -> MIDI pipeline for Logic Pro cover arrangements.",
)
console = Console()


def _check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        console.print(
            "[red]ffmpeg not found on PATH.[/red] "
            "Install with `brew install ffmpeg` (macOS) or `apt install ffmpeg` (Linux)."
        )
        raise typer.Exit(code=2)


def _parse_steps(raw: str) -> list[str]:
    requested = [s.strip() for s in raw.split(",") if s.strip()]
    unknown = [s for s in requested if s not in ALL_STEPS]
    if unknown:
        console.print(f"[red]Unknown steps:[/red] {unknown}. Valid: {list(ALL_STEPS)}")
        raise typer.Exit(code=2)
    # Preserve canonical order regardless of user order.
    return [s for s in ALL_STEPS if s in requested]


@app.command()
def main(
    playlist_url: str = typer.Argument(..., help="Spotify playlist URL."),
    output: Path = typer.Option(
        Path("./output"), "--output", "-o", help="Output root directory."
    ),
    songs_limit: Optional[int] = typer.Option(
        None, "--songs-limit", help="Process only the first N tracks (for testing)."
    ),
    steps: str = typer.Option(
        ",".join(ALL_STEPS),
        "--steps",
        help=f"Comma-separated subset of {list(ALL_STEPS)}.",
    ),
    force: bool = typer.Option(
        False, "--force", help="Re-run even if outputs already exist."
    ),
    skip_melodic: bool = typer.Option(
        False, "--skip-melodic", help="Skip basic-pitch transcription (vocals/bass/other)."
    ),
    skip_drums: bool = typer.Option(
        False, "--skip-drums", help="Skip drum transcription."
    ),
) -> None:
    """Run the pipeline against a Spotify playlist URL."""

    step_list = _parse_steps(steps)
    _check_ffmpeg()

    storage = LocalStorage(output)
    console.print(f"[bold]Output root:[/bold] {storage.root}")

    try:
        results = run_pipeline(
            playlist_url,
            storage,
            console,
            steps=step_list,
            songs_limit=songs_limit,
            force=force,
            skip_melodic=skip_melodic,
            skip_drums=skip_drums,
        )
    except MissingDependencyError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=2) from e

    # Compact summary table.
    if results:
        console.print()
        console.print(f"[bold green]Processed {len(results)} song(s).[/bold green]")
        for r in results:
            status = (
                "[red]download failed[/red]"
                if "download_error" in r
                else "[red]fatal error[/red]"
                if "fatal_error" in r
                else "[green]ok[/green]"
            )
            console.print(
                f"  - {r.get('artist','?')} - {r.get('title','?')}: {status} "
                f"(tempo={r.get('tempo_bpm','?')}, key={r.get('estimated_key','?')})"
            )

    sys.exit(0)


if __name__ == "__main__":
    app()
