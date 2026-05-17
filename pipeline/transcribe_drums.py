"""Step 3b: drum stem -> MIDI via ADTOF, with documented fallback chain.

Engines tried (in order):
  1. ADTOF-pytorch (https://github.com/MZehren/ADTOF) — primary.
  2. Magenta Onsets-and-Frames drum model — documented fallback per spec.
  3. librosa onset detection across filtered bands — last-resort heuristic.
     Marked clearly in the output metadata so downstream knows quality is low.

All engines emit a list of DrumHit objects which we then write through
`drum_midi_mapping.drum_hits_to_midi_notes`. This keeps the GM mapping in
one place — see `drum_midi_mapping.py` to customize for different samplers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pretty_midi

from pipeline.drum_midi_mapping import (
    DrumHit,
    canonicalize_label,
    drum_hits_to_midi_notes,
)
from pipeline.errors import MissingDependencyError
from pipeline.storage import Storage


def _try_adtof(audio_path: Path) -> list[DrumHit] | None:
    try:
        # ADTOF's public entry point varies across forks. Try the common one.
        from adtof.io import textReaders  # noqa: F401  type: ignore
        from adtof.model.model import Model  # type: ignore
    except ImportError:
        return None

    try:
        model = Model.modelFromName("crnn-all")  # ADTOF's pretrained 5-class CRNN
        events = model.predict(str(audio_path))
        # ADTOF returns a list of (time_s, pitch_or_label). Normalize.
        hits: list[DrumHit] = []
        for ev in events:
            if isinstance(ev, dict):
                t = float(ev["time"])
                label = canonicalize_label(str(ev.get("label") or ev.get("class")))
            else:
                t, raw = float(ev[0]), str(ev[1])
                label = canonicalize_label(raw)
            hits.append(DrumHit(time_s=t, label=label, velocity=100))
        return hits
    except Exception as e:
        # ADTOF imports but fails at inference — surface to caller via None.
        # Caller decides whether to fall back.
        print(f"  ADTOF inference failed ({e}); falling back to next engine.")
        return None


def _try_magenta(audio_path: Path) -> list[DrumHit] | None:
    try:
        from magenta.models.onsets_frames_transcription import (  # type: ignore
            audio_label_data_utils,  # noqa: F401
            train_util,  # noqa: F401
        )
        import note_seq  # noqa: F401  type: ignore
    except ImportError:
        return None
    # Magenta's drum inference is a multi-step TF-1 process; documenting
    # but not implementing here keeps the POC scope tight. If you get here
    # in practice (ADTOF unusable), see README's "Optional upgrade paths".
    print("  Magenta installed but the OAF drum inference path is not wired up "
          "in this POC. Set up per README, or use librosa fallback for now.")
    return None


def _librosa_band_onset_fallback(audio_path: Path) -> list[DrumHit]:
    """Cheap, accuracy-limited drum transcription via librosa onset detection.

    Splits the drum stem into 3 frequency bands (kick, snare, hi-hat) using
    butterworth filters, runs `librosa.onset.onset_detect` on each, and
    assigns the canonical label. We do not attempt to detect toms or cymbals
    in this fallback — they show up under "snare" or "hihat" depending on
    energy, which is wrong but predictable.

    Mark the output as low-quality in metadata so the user knows to swap
    in a real drum-transcription model before serious use.
    """
    import librosa
    from scipy.signal import butter, sosfilt

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)

    def band_onsets(low_hz: float, high_hz: float) -> np.ndarray:
        nyq = sr / 2
        low = max(low_hz / nyq, 1e-4)
        high = min(high_hz / nyq, 0.999)
        sos = butter(4, [low, high], btype="band", output="sos")
        filtered = sosfilt(sos, y).astype(np.float32)
        return librosa.onset.onset_detect(
            y=filtered,
            sr=sr,
            units="time",
            backtrack=False,
            delta=0.1,
        )

    hits: list[DrumHit] = []
    for t in band_onsets(20, 120):
        hits.append(DrumHit(time_s=float(t), label="kick", velocity=110))
    for t in band_onsets(150, 700):
        hits.append(DrumHit(time_s=float(t), label="snare", velocity=100))
    for t in band_onsets(4000, 10000):
        hits.append(DrumHit(time_s=float(t), label="hihat", velocity=90))

    hits.sort(key=lambda h: h.time_s)
    return hits


def _write_drum_midi(hits: list[DrumHit], out_path: Path) -> None:
    """Write hits to a Type-1 MIDI with one drum track on channel 10."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    # is_drum=True puts the instrument on MIDI channel 10 (drums) when saved.
    drum_inst = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
    for pitch, start, end, vel in drum_hits_to_midi_notes(hits):
        drum_inst.notes.append(
            pretty_midi.Note(velocity=vel, pitch=pitch, start=start, end=end)
        )
    pm.instruments.append(drum_inst)
    pm.write(str(out_path))


def transcribe_drums(
    input_stem_key: str,
    output_midi_key: str,
    storage: Storage,
    force: bool = False,
) -> dict[str, Any]:
    """Transcribe drum stem -> GM-mapped MIDI.

    Tries ADTOF first, then Magenta, then a librosa-band-onset fallback.
    Returns a dict including which engine was used so it can be recorded in
    metadata.json and shown to the user.
    """
    if storage.exists(output_midi_key) and not force:
        return {
            "midi_key": output_midi_key,
            "skipped": True,
            "engine": "(skipped — output already existed)",
        }

    input_path = storage.local_path(input_stem_key)
    if not input_path.exists():
        raise FileNotFoundError(f"Drum stem missing: {input_stem_key}")

    engine: str
    hits = _try_adtof(input_path)
    if hits is not None:
        engine = "adtof-crnn-all"
    else:
        hits = _try_magenta(input_path)
        if hits is not None:
            engine = "magenta-oaf-drums"
        else:
            hits = _librosa_band_onset_fallback(input_path)
            engine = "librosa-band-onsets (low quality fallback)"

    out_path = storage.local_path(output_midi_key)
    _write_drum_midi(hits, out_path)

    return {
        "midi_key": output_midi_key,
        "skipped": False,
        "engine": engine,
        "n_hits": len(hits),
    }
