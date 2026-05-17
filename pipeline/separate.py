"""Step 2: demucs 4-stem separation with htdemucs_ft on MPS when available."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from pipeline.errors import MissingDependencyError
from pipeline.storage import Storage

STEM_NAMES = ("vocals", "drums", "bass", "other")


def _pick_device() -> str:
    """Prefer MPS (Apple Silicon) -> CUDA -> CPU. demucs reads device via arg."""
    try:
        import torch
    except ImportError as e:
        raise MissingDependencyError(
            "torch",
            "Install with `uv sync`. PyTorch with MPS requires macOS 12.3+.",
        ) from e
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _ensure_demucs() -> Any:
    try:
        from demucs import separate as demucs_separate  # type: ignore[import-not-found]
        return demucs_separate
    except ImportError as e:
        raise MissingDependencyError(
            "demucs",
            "Install with `uv sync` or `uv pip install demucs`.",
        ) from e


def separate(
    input_audio_key: str,
    output_prefix: str,
    storage: Storage,
    force: bool = False,
    model: str = "htdemucs_ft",
    device: str | None = None,
) -> dict[str, Any]:
    """Split `input_audio_key` into 4 stems under `{output_prefix}/stems/`.

    Signature follows the project convention exactly:
        separate(input_audio_key, output_prefix, storage) -> dict.

    Stems written:
        {output_prefix}/stems/vocals.wav
        {output_prefix}/stems/drums.wav
        {output_prefix}/stems/bass.wav
        {output_prefix}/stems/other.wav

    Idempotent: if all four stems already exist and force=False, skip.

    htdemucs_ft is a fine-tuned 4-stem model — strictly better quality than
    the base htdemucs at the cost of ~4x runtime. We run it sequentially
    across songs because it's GPU-bound.
    """
    stem_keys = {name: f"{output_prefix}/stems/{name}.wav" for name in STEM_NAMES}

    if not force and all(storage.exists(k) for k in stem_keys.values()):
        return {"stems": stem_keys, "skipped": True, "model": model}

    demucs_separate = _ensure_demucs()
    chosen_device = device or _pick_device()

    input_path = storage.local_path(input_audio_key)
    if not input_path.exists():
        raise FileNotFoundError(f"Input audio missing: {input_audio_key}")

    # demucs.separate.main parses argv and writes to `-o <dir>`. It creates
    # `<out>/<model>/<input_stem_name>/{vocals,drums,bass,other}.wav`.
    # We hand it a fresh subdir we control, then rename into place.
    with storage.workdir(f"{output_prefix}/stems") as stems_dir:
        # demucs writes a nested directory; aim it at a tmp inside stems_dir.
        tmp_root = stems_dir / "_demucs_tmp"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        tmp_root.mkdir(parents=True, exist_ok=True)

        args = [
            "--name", model,
            "-o", str(tmp_root),
            "-d", chosen_device,
            "--filename", "{stem}.{ext}",
            str(input_path),
        ]
        # demucs.separate.main raises SystemExit on errors; let it propagate.
        demucs_separate.main(args)

        # Find the nested folder demucs produced. With --filename "{stem}.{ext}"
        # demucs writes to {tmp_root}/{model}/{stem}.wav.
        produced = tmp_root / model
        if not produced.exists():
            # Older demucs versions may use a different nesting. Search.
            candidates = list(tmp_root.rglob("vocals.wav"))
            if not candidates:
                raise RuntimeError(
                    f"demucs produced no output under {tmp_root}. Check stderr above."
                )
            produced = candidates[0].parent

        for name in STEM_NAMES:
            src = produced / f"{name}.wav"
            if not src.exists():
                raise RuntimeError(f"demucs did not produce {name}.wav at {src}")
            dst = storage.local_path(stem_keys[name])
            shutil.move(str(src), dst)

        shutil.rmtree(tmp_root, ignore_errors=True)

    return {"stems": stem_keys, "skipped": False, "model": model, "device": chosen_device}
