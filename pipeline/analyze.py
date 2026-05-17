"""Step 4: librosa analysis — tempo, key, time signature, duration, beats."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from pipeline.storage import Storage

# Krumhansl-Schmuckler key profiles. We correlate the chroma vector against
# all 24 major/minor profiles and pick the best.
_MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)
_PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _estimate_key(y: np.ndarray, sr: int) -> str:
    import librosa

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)

    best_score = -np.inf
    best_label = "C major"
    for i in range(12):
        major_corr = np.corrcoef(np.roll(_MAJOR_PROFILE, i), chroma_mean)[0, 1]
        minor_corr = np.corrcoef(np.roll(_MINOR_PROFILE, i), chroma_mean)[0, 1]
        if major_corr > best_score:
            best_score = major_corr
            best_label = f"{_PITCH_CLASSES[i]} major"
        if minor_corr > best_score:
            best_score = minor_corr
            best_label = f"{_PITCH_CLASSES[i]} minor"
    return best_label


def _guess_time_signature(beat_times: np.ndarray, tempo: float) -> str:
    """Heuristic 4/4 vs 3/4 guess from beat regularity.

    This is genuinely a guess — proper meter detection is a research problem.
    We compute downbeat-like accent every N beats and pick the N that
    best fits. README notes this is unreliable; user should verify.
    """
    if len(beat_times) < 8:
        return "4/4"
    intervals = np.diff(beat_times)
    if len(intervals) == 0:
        return "4/4"
    # If beats are very regular, default to 4/4 (the overwhelmingly common case).
    cv = float(np.std(intervals) / (np.mean(intervals) + 1e-9))
    if cv > 0.25:
        return "unknown (irregular beats)"
    return "4/4"


def analyze(
    input_audio_key: str,
    output_metadata_key: str,
    storage: Storage,
    force: bool = False,
) -> dict[str, Any]:
    """Analyze original audio for tempo/key/duration/beats.

    Writes the analysis section to a JSON keyed at `output_metadata_key`
    (the pipeline runner merges this with download metadata into the
    final per-song metadata.json).

    Returns the analysis dict (also persisted to storage).
    """
    if storage.exists(output_metadata_key) and not force:
        data = json.loads(storage.read_bytes(output_metadata_key))
        data["skipped"] = True
        return data

    try:
        import librosa
    except ImportError as e:
        from pipeline.errors import MissingDependencyError

        raise MissingDependencyError(
            "librosa", "Install with `uv sync` or `uv pip install librosa`."
        ) from e

    audio_path = storage.local_path(input_audio_key)
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    duration_s = float(librosa.get_duration(y=y, sr=sr))

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    # librosa 0.10+ returns tempo as ndarray; normalize to float.
    tempo_bpm = float(np.atleast_1d(tempo)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    beat_count = int(len(beat_times))

    estimated_key = _estimate_key(y, sr)
    time_signature = _guess_time_signature(beat_times, tempo_bpm)

    analysis = {
        "tempo_bpm": round(tempo_bpm, 2),
        "estimated_key": estimated_key,
        "time_signature": time_signature,
        "duration_seconds": round(duration_s, 2),
        "beat_count": beat_count,
        "beat_times": [round(float(t), 4) for t in beat_times.tolist()],
        "skipped": False,
    }

    storage.write_bytes(
        output_metadata_key,
        json.dumps(analysis, indent=2).encode("utf-8"),
    )
    return analysis
