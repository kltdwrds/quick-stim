"""Storage abstraction for pipeline outputs.

The pipeline never touches the filesystem directly — every read, write,
existence check, and listing goes through a Storage implementation.
This lets us swap in an R2/S3-backed Storage later without changing any
step code.

Keys are forward-slash-separated, e.g. "Daft Punk - One More Time/stems/bass.wav".
They are *not* filesystem paths; LocalStorage maps them to paths under a
configurable root.

For external tools that need a filesystem path (spotdl, demucs, basic-pitch,
ADTOF — none of which accept file-like objects for output), Storage exposes
`local_path(key)`. For LocalStorage this is a direct path; a future cloud
Storage would materialize the key into a temp file and persist on close
(via a context manager — see `workdir` below).
"""

from __future__ import annotations

import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable


@runtime_checkable
class Storage(Protocol):
    """Minimal storage interface used by every pipeline step."""

    def read_bytes(self, key: str) -> bytes: ...
    def write_bytes(self, key: str, data: bytes) -> None: ...
    def exists(self, key: str) -> bool: ...
    def list(self, prefix: str) -> list[str]: ...

    # Escape hatch for external tools that demand a real filesystem path.
    # For LocalStorage this is the path under the output root. A future
    # R2Storage would override workdir() to download-on-enter, upload-on-exit.
    def local_path(self, key: str) -> Path: ...

    @contextmanager
    def workdir(self, prefix: str) -> Iterator[Path]:
        """Yield a local directory whose contents map to `prefix` in storage.

        For LocalStorage this is a no-op (the directory IS the storage).
        Cloud impls would download prefix contents on enter and upload
        modified files on exit.
        """
        ...


class LocalStorage:
    """Local filesystem storage. Keys map directly to paths under `root`."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        # Guard against path traversal via "../" in keys. Keys are pipeline-internal,
        # but they include song titles from Spotify which we don't fully trust.
        path = (self.root / key).resolve()
        if not str(path).startswith(str(self.root)):
            raise ValueError(f"Key escapes storage root: {key!r}")
        return path

    def read_bytes(self, key: str) -> bytes:
        return self._resolve(key).read_bytes()

    def write_bytes(self, key: str, data: bytes) -> None:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def exists(self, key: str) -> bool:
        return self._resolve(key).exists()

    def list(self, prefix: str) -> list[str]:
        base = self._resolve(prefix)
        if not base.exists():
            return []
        if base.is_file():
            return [prefix]
        out: list[str] = []
        for p in base.rglob("*"):
            if p.is_file():
                rel = p.relative_to(self.root).as_posix()
                out.append(rel)
        return sorted(out)

    def local_path(self, key: str) -> Path:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @contextmanager
    def workdir(self, prefix: str) -> Iterator[Path]:
        path = self._resolve(prefix)
        path.mkdir(parents=True, exist_ok=True)
        yield path

    # Convenience used by tests / CLI summaries; not part of the protocol.
    def copy_in(self, src: Path, key: str) -> None:
        dst = self.local_path(key)
        shutil.copyfile(src, dst)
