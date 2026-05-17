"""Step 1: download Spotify playlist tracks via spotdl (YouTube matching).

We treat the download as: for each Spotify track, produce
    "<artist> - <title>/original.mp3"
under the storage root. spotdl provides the YouTube matching + ffmpeg
re-encoding; we just wrap its API.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline.errors import DownloadError, MissingDependencyError
from pipeline.storage import Storage


@dataclass
class TrackInfo:
    """Minimal track metadata extracted from spotdl before download."""

    title: str
    artist: str
    spotify_url: str
    duration_s: float
    song_prefix: str  # storage key prefix: "Artist - Title"


_INVALID_PATH_CHARS = re.compile(r"[^\w\s.,()&'\-]+", re.UNICODE)


def sanitize_for_path(s: str, max_len: int = 80) -> str:
    """Make a string safe for use as a filesystem path segment.

    spotdl has its own sanitizer but we use this for the storage *key*,
    which is decoupled from spotdl's output template.
    """
    s = s.strip()
    s = _INVALID_PATH_CHARS.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_len:
        s = s[:max_len].rstrip()
    return s or "untitled"


def _import_spotdl() -> Any:
    try:
        from spotdl import Spotdl  # type: ignore[import-not-found]
        from spotdl.types.song import Song  # noqa: F401  (sanity check)
        return Spotdl
    except ImportError as e:
        raise MissingDependencyError(
            "spotdl",
            "Install with `uv sync` or `uv pip install spotdl`.",
        ) from e


def list_playlist_tracks(
    playlist_url: str,
    songs_limit: int | None = None,
) -> tuple[Any, list[Any]]:
    """Query Spotify (via spotdl) for the playlist's tracks.

    Returns (spotdl_client, [Song, ...]). spotdl handles auth with public
    client credentials embedded in the package.
    """
    Spotdl = _import_spotdl()
    # Public client creds shipped with spotdl. No user auth required for
    # reading public playlists.
    client = Spotdl(
        client_id="5f573c9620494bae87890c0f08a60293",
        client_secret="212476d9b0f3472eaa762d90b19b0ba8",
        no_cache=True,
        headless=True,
    )
    songs = client.search([playlist_url])
    if songs_limit is not None:
        songs = songs[:songs_limit]
    return client, songs


def track_info_from_song(song: Any) -> TrackInfo:
    """Build our internal TrackInfo from a spotdl Song object."""
    artist = sanitize_for_path(song.artist or "Unknown Artist")
    title = sanitize_for_path(song.name or "Untitled")
    return TrackInfo(
        title=song.name or "Untitled",
        artist=song.artist or "Unknown Artist",
        spotify_url=song.url,
        duration_s=float(song.duration or 0.0),
        song_prefix=f"{artist} - {title}",
    )


def download(
    spotify_song: Any,
    output_prefix: str,
    storage: Storage,
    spotdl_client: Any,
    force: bool = False,
) -> dict[str, Any]:
    """Download a single Spotify song into `{output_prefix}/original.mp3`.

    Signature shape matches the project convention: explicit input
    (spotify_song + spotdl_client), explicit output prefix, Storage handle.

    Idempotent: if `{output_prefix}/original.mp3` already exists and
    force=False, skip the actual download.

    Returns a metadata dict for the per-song metadata.json:
        {
          "title", "artist", "spotify_url", "duration_seconds",
          "original_audio_key", "skipped"
        }
    """
    info = track_info_from_song(spotify_song)
    audio_key = f"{output_prefix}/original.mp3"

    if storage.exists(audio_key) and not force:
        return {
            "title": info.title,
            "artist": info.artist,
            "spotify_url": info.spotify_url,
            "duration_seconds": info.duration_s,
            "original_audio_key": audio_key,
            "skipped": True,
        }

    # spotdl writes to a templated output path. We give it a fresh temp dir
    # alongside the storage location, then move the mp3 into our key.
    with storage.workdir(output_prefix) as song_dir:
        tmp_dir = song_dir / "_spotdl_tmp"
        tmp_dir.mkdir(exist_ok=True)

        # spotdl's per-song API: Spotdl.download(Song) returns (Song, Path|None)
        try:
            # Force output to our temp dir via the settings.
            spotdl_client.downloader.settings["output"] = str(tmp_dir / "{title}.{output-ext}")
            spotdl_client.downloader.settings["format"] = "mp3"
            spotdl_client.downloader.settings["overwrite"] = "force" if force else "skip"
            _song, downloaded_path = spotdl_client.download(spotify_song)
        except Exception as e:  # spotdl raises a grab-bag of exceptions
            raise DownloadError(
                f"spotdl failed for {info.artist} - {info.title}: {e}"
            ) from e

        if downloaded_path is None or not Path(downloaded_path).exists():
            raise DownloadError(
                f"spotdl could not match {info.artist} - {info.title} on YouTube."
            )

        final = storage.local_path(audio_key)
        shutil.move(str(downloaded_path), final)
        # Clean up spotdl's temp dir.
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return {
        "title": info.title,
        "artist": info.artist,
        "spotify_url": info.spotify_url,
        "duration_seconds": info.duration_s,
        "original_audio_key": audio_key,
        "skipped": False,
    }
