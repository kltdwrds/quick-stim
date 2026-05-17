"""Per-song pipeline orchestration.

This is the only module that knows about the whole pipeline sequence.
Each step function in download/separate/transcribe_*/analyze is independent
and can be invoked alone (suitable for future Modal/container task split).
"""

from __future__ import annotations

import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Sequence

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from pipeline import analyze as analyze_mod
from pipeline import download as download_mod
from pipeline import separate as separate_mod
from pipeline import transcribe_drums as drums_mod
from pipeline import transcribe_melodic as melodic_mod
from pipeline.errors import DownloadError, MissingDependencyError, PipelineError
from pipeline.storage import Storage

ALL_STEPS = ("download", "separate", "transcribe", "analyze")
MELODIC_STEMS = ("bass", "other", "vocals")


def _log(meta: dict[str, Any], msg: str) -> None:
    meta.setdefault("processing_log", []).append(
        {"t": round(time.time(), 2), "msg": msg}
    )


def process_song(
    spotify_song: Any,
    spotdl_client: Any,
    storage: Storage,
    console: Console,
    steps: Sequence[str],
    *,
    force: bool = False,
    skip_melodic: bool = False,
    skip_drums: bool = False,
    parallel_transcribe: bool = True,
) -> dict[str, Any]:
    """Run all enabled steps for a single song; return the per-song metadata."""

    info = download_mod.track_info_from_song(spotify_song)
    song_prefix = info.song_prefix
    metadata_key = f"{song_prefix}/metadata.json"

    # Start from any prior metadata.json on disk so re-runs preserve history.
    if storage.exists(metadata_key) and not force:
        try:
            meta: dict[str, Any] = json.loads(storage.read_bytes(metadata_key))
        except json.JSONDecodeError:
            meta = {}
    else:
        meta = {}

    meta.setdefault("title", info.title)
    meta.setdefault("artist", info.artist)
    meta.setdefault("spotify_url", info.spotify_url)
    meta.setdefault(
        "transcription_notes",
        {
            "drums": "ADTOF 5-class, GM-mapped",
            "vocals": "Reference only — pitch contour and phrasing scaffolding",
            "other": "Polyphonic, expect manual cleanup",
        },
    )
    meta.setdefault("processing_log", [])

    # Step 1: download ---------------------------------------------------
    if "download" in steps:
        try:
            r = download_mod.download(
                spotify_song, song_prefix, storage, spotdl_client, force=force
            )
            meta["original_audio_key"] = r["original_audio_key"]
            meta["duration_seconds"] = r["duration_seconds"]
            _log(meta, f"download: {'skipped' if r['skipped'] else 'ok'}")
        except DownloadError as e:
            meta["download_error"] = str(e)
            _log(meta, f"download FAILED: {e}")
            console.print(f"[red]  download failed:[/red] {e}")
            _persist_meta(storage, metadata_key, meta)
            return meta

    audio_key = meta.get("original_audio_key", f"{song_prefix}/original.mp3")

    # Step 2: separate ---------------------------------------------------
    if "separate" in steps:
        try:
            r = separate_mod.separate(audio_key, song_prefix, storage, force=force)
            # Output schema uses short filenames; storage keeps full keys internally.
            meta["stems"] = [f"{name}.wav" for name in r["stems"].keys()]
            meta["_stem_keys"] = r["stems"]  # private: keys (not just names) for next step
            _log(meta, f"separate: {'skipped' if r['skipped'] else 'ok'} ({r.get('device','?')})")
        except PipelineError as e:
            meta["separate_error"] = str(e)
            _log(meta, f"separate FAILED: {e}")
            console.print(f"[red]  separate failed:[/red] {e}")
            _persist_meta(storage, metadata_key, meta)
            return meta

    stem_keys: dict[str, str] = meta.get("_stem_keys") or {
        name: f"{song_prefix}/stems/{name}.wav"
        for name in ("vocals", "drums", "bass", "other")
    }

    # Step 3: transcribe -------------------------------------------------
    if "transcribe" in steps:
        midi_files: dict[str, str] = meta.get("midi_files", {}) or {}

        melodic_tasks: list[tuple[str, str, str]] = []  # (stem_kind, in_key, out_key)
        if not skip_melodic:
            for stem in MELODIC_STEMS:
                in_key = stem_keys[stem]
                out_key = f"{song_prefix}/midi/{stem}.mid"
                melodic_tasks.append((stem, in_key, out_key))

        def _run_melodic(task: tuple[str, str, str]) -> tuple[str, dict[str, Any]]:
            stem_kind, in_key, out_key = task
            return stem_kind, melodic_mod.transcribe_melodic(
                in_key, out_key, storage, stem_kind=stem_kind, force=force
            )

        if melodic_tasks:
            if parallel_transcribe and len(melodic_tasks) > 1:
                with ThreadPoolExecutor(max_workers=len(melodic_tasks)) as ex:
                    futures = {ex.submit(_run_melodic, t): t for t in melodic_tasks}
                    for fut in as_completed(futures):
                        try:
                            stem_kind, r = fut.result()
                            midi_files[stem_kind] = f"{stem_kind}.mid"
                            _log(meta, f"transcribe[{stem_kind}]: {'skipped' if r['skipped'] else 'ok'}")
                        except Exception as e:
                            t = futures[fut]
                            _log(meta, f"transcribe[{t[0]}] FAILED: {e}")
                            console.print(f"[red]  transcribe[{t[0]}] failed:[/red] {e}")
            else:
                for task in melodic_tasks:
                    try:
                        stem_kind, r = _run_melodic(task)
                        midi_files[stem_kind] = f"{stem_kind}.mid"
                        _log(meta, f"transcribe[{stem_kind}]: {'skipped' if r['skipped'] else 'ok'}")
                    except Exception as e:
                        _log(meta, f"transcribe[{task[0]}] FAILED: {e}")
                        console.print(f"[red]  transcribe[{task[0]}] failed:[/red] {e}")

        if not skip_drums:
            in_key = stem_keys["drums"]
            out_key = f"{song_prefix}/midi/drums.mid"
            try:
                r = drums_mod.transcribe_drums(in_key, out_key, storage, force=force)
                midi_files["drums"] = "drums.mid"
                meta["drum_engine"] = r["engine"]
                _log(
                    meta,
                    f"transcribe[drums]: {'skipped' if r['skipped'] else 'ok'} "
                    f"(engine={r['engine']})",
                )
            except Exception as e:
                _log(meta, f"transcribe[drums] FAILED: {e}")
                console.print(f"[red]  transcribe[drums] failed:[/red] {e}")

        meta["midi_files"] = midi_files

    # Step 4: analyze ----------------------------------------------------
    if "analyze" in steps:
        try:
            analysis_key = f"{song_prefix}/_analysis.json"
            r = analyze_mod.analyze(audio_key, analysis_key, storage, force=force)
            meta["tempo_bpm"] = r["tempo_bpm"]
            meta["estimated_key"] = r["estimated_key"]
            meta["time_signature"] = r["time_signature"]
            meta["duration_seconds"] = r["duration_seconds"]
            meta["beat_count"] = r["beat_count"]
            meta["beat_times"] = r["beat_times"]
            _log(meta, f"analyze: {'skipped' if r.get('skipped') else 'ok'}")
        except PipelineError as e:
            meta["analyze_error"] = str(e)
            _log(meta, f"analyze FAILED: {e}")
            console.print(f"[red]  analyze failed:[/red] {e}")

    # Strip internal keys before persisting.
    meta.pop("_stem_keys", None)

    _persist_meta(storage, metadata_key, meta)
    return meta


def _persist_meta(storage: Storage, key: str, meta: dict[str, Any]) -> None:
    storage.write_bytes(key, json.dumps(meta, indent=2).encode("utf-8"))


def run_pipeline(
    playlist_url: str,
    storage: Storage,
    console: Console,
    *,
    steps: Sequence[str],
    songs_limit: int | None,
    force: bool,
    skip_melodic: bool,
    skip_drums: bool,
) -> list[dict[str, Any]]:
    """Top-level entry: fetch playlist, then process each song."""

    console.print(f"[cyan]Fetching playlist:[/cyan] {playlist_url}")
    try:
        client, songs = download_mod.list_playlist_tracks(playlist_url, songs_limit)
    except MissingDependencyError as e:
        console.print(f"[red]{e}[/red]")
        raise

    if not songs:
        console.print("[yellow]No songs found in playlist.[/yellow]")
        return []

    console.print(f"[green]Found {len(songs)} song(s).[/green] Steps: {', '.join(steps)}")

    results: list[dict[str, Any]] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Processing songs", total=len(songs))
        for song in songs:
            info = download_mod.track_info_from_song(song)
            progress.update(task_id, description=f"  {info.artist} - {info.title}")
            try:
                meta = process_song(
                    song,
                    client,
                    storage,
                    console,
                    steps,
                    force=force,
                    skip_melodic=skip_melodic,
                    skip_drums=skip_drums,
                )
                results.append(meta)
            except Exception as e:
                # Per-song failures shouldn't kill the playlist run.
                console.print(f"[red]  Unhandled error on {info.song_prefix}:[/red] {e}")
                console.print(traceback.format_exc())
                results.append(
                    {
                        "title": info.title,
                        "artist": info.artist,
                        "spotify_url": info.spotify_url,
                        "fatal_error": str(e),
                    }
                )
            progress.advance(task_id)

    return results
