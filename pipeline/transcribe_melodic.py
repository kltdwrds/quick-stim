"""Step 3a: basic-pitch transcription for vocals / bass / other stems.

We use basic-pitch's `predict()` function (not its CLI) so we can tune
parameters per stem type and write the MIDI through Storage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pipeline.errors import MissingDependencyError
from pipeline.storage import Storage


# Per-stem parameter sets. basic-pitch's defaults are tuned for instrumental
# audio; vocals benefit from a lower confidence threshold (more permissive)
# and a shorter minimum note length (catches consonant attacks and ornaments).
STEM_PARAMS: dict[str, dict[str, Any]] = {
    "bass": {
        # Default basic-pitch params. Monophonic bass transcribes well out of the box.
    },
    "other": {
        # Default basic-pitch params. Polyphonic; ~60-70% accuracy expected.
    },
    "vocals": {
        # User-tuned for melodic vocal reference:
        # - lower confidence threshold catches softer/legato notes
        # - shorter min note length captures phrasing detail
        # - constrain pitch range to typical sung range (C2-C6)
        "onset_threshold": 0.5,  # basic-pitch parameter name
        "frame_threshold": 0.3,  # the "model_confidence_threshold" in the spec maps to frame_threshold
        "minimum_note_length": 58.0,  # milliseconds
        "minimum_frequency": 65.41,  # C2 == MIDI 36
        "maximum_frequency": 1046.50,  # C6 == MIDI 84
    },
}


def _ensure_basic_pitch() -> tuple[Any, Any]:
    try:
        from basic_pitch.inference import predict  # type: ignore[import-not-found]
        from basic_pitch import ICASSP_2022_MODEL_PATH  # type: ignore[import-not-found]
        return predict, ICASSP_2022_MODEL_PATH
    except ImportError as e:
        raise MissingDependencyError(
            "basic_pitch",
            "Install with `uv sync` or `uv pip install basic-pitch`.",
        ) from e


def transcribe_melodic(
    input_stem_key: str,
    output_midi_key: str,
    storage: Storage,
    stem_kind: str,
    force: bool = False,
) -> dict[str, Any]:
    """Transcribe a melodic stem to MIDI.

    Signature: explicit input + output keys + storage + stem_kind so the
    function is self-contained and trivially re-invocable as a Modal /
    container task.

    stem_kind in {"bass", "other", "vocals"} selects which parameter set
    to use (see STEM_PARAMS above).
    """
    if stem_kind not in STEM_PARAMS:
        raise ValueError(
            f"stem_kind={stem_kind!r} not in {set(STEM_PARAMS)}. "
            "For drums, use pipeline.transcribe_drums instead."
        )

    if storage.exists(output_midi_key) and not force:
        return {"midi_key": output_midi_key, "skipped": True, "stem_kind": stem_kind}

    predict, _model_path = _ensure_basic_pitch()
    params = STEM_PARAMS[stem_kind]

    input_path = storage.local_path(input_stem_key)
    if not input_path.exists():
        raise FileNotFoundError(f"Stem audio missing: {input_stem_key}")

    # basic_pitch.predict returns (model_output_dict, midi_data, note_events).
    # We pass through any params that match its signature; basic-pitch is
    # tolerant of extra kwargs in recent versions but we filter to known ones.
    valid_kwargs = {
        "onset_threshold",
        "frame_threshold",
        "minimum_note_length",
        "minimum_frequency",
        "maximum_frequency",
        "multiple_pitch_bends",
        "melodia_trick",
    }
    kwargs = {k: v for k, v in params.items() if k in valid_kwargs}

    _model_out, midi_data, _notes = predict(str(input_path), **kwargs)

    out_path = storage.local_path(output_midi_key)
    midi_data.write(str(out_path))

    return {
        "midi_key": output_midi_key,
        "skipped": False,
        "stem_kind": stem_kind,
        "params_used": kwargs,
    }
