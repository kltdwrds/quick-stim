"""Maps ADTOF's 5-class drum output to General MIDI note numbers.

ADTOF (Zehren et al., https://github.com/MZehren/ADTOF) emits onsets in 5
classes derived from the consensus across drum transcription datasets:

    kick      - bass drum
    snare     - snare drum
    hi-hat    - any closed/open hi-hat
    toms      - any tom (we collapse to low-tom by default)
    cymbals   - ride / crash (we collapse to crash by default)

General MIDI percussion (channel 10) note assignments we use:

    36  C1   Acoustic Bass Drum (kick)
    38  D1   Acoustic Snare
    42  F#1  Closed Hi-Hat
    45  A1   Low Tom
    49  C#2  Crash Cymbal 1

Why these specific GM notes:
- 36/38/42 are universal across every drum sampler I've used (Logic's Drum
  Kit Designer, Superior Drummer, BFD, EZdrummer, Native Drumlab).
- Tom assignment is ambiguous in 5-class — most kits have 3-4 toms (low/mid/hi/floor)
  and ADTOF doesn't split them. Picking low-tom (45) as a single bucket keeps it
  predictable; you can re-route in Logic's MIDI environment if you want spread.
- Crash (49) is more common than ride (51) in pop/rock covers, so it's the
  safer default. Manually re-map ride-heavy songs in Logic.

To customize for a specific sampler (e.g. SSD5 expects 36/38/42 too but uses
different velocity layers), edit GM_DRUM_MAP below or build an alternate map
and pass it into `drums_events_to_midi`.
"""

from __future__ import annotations

from dataclasses import dataclass

# ADTOF 5-class label -> General MIDI percussion note.
GM_DRUM_MAP: dict[str, int] = {
    "kick": 36,
    "snare": 38,
    "hihat": 42,
    "tom": 45,
    "cymbal": 49,
}

# ADTOF may report class names with variations across versions. Normalize.
ADTOF_LABEL_ALIASES: dict[str, str] = {
    # canonical -> canonical
    "kick": "kick",
    "snare": "snare",
    "hihat": "hihat",
    "tom": "tom",
    "cymbal": "cymbal",
    # known variants
    "bd": "kick",
    "bass_drum": "kick",
    "sd": "snare",
    "hh": "hihat",
    "hi_hat": "hihat",
    "hat": "hihat",
    "toms": "tom",
    "tt": "tom",
    "cymbals": "cymbal",
    "cy": "cymbal",
    "cr": "cymbal",
    "crash": "cymbal",
    "ride": "cymbal",
}


@dataclass(frozen=True)
class DrumHit:
    """One drum onset. `time_s` is absolute seconds from start of audio."""

    time_s: float
    label: str  # canonical: kick / snare / hihat / tom / cymbal
    velocity: int = 100  # 0-127 GM velocity


def canonicalize_label(raw: str) -> str:
    """Map a raw ADTOF label to one of {kick, snare, hihat, tom, cymbal}."""
    key = raw.strip().lower().replace("-", "_")
    if key in ADTOF_LABEL_ALIASES:
        return ADTOF_LABEL_ALIASES[key]
    raise ValueError(
        f"Unknown drum label {raw!r}. "
        f"Add it to ADTOF_LABEL_ALIASES in drum_midi_mapping.py."
    )


def drum_hits_to_midi_notes(
    hits: list[DrumHit],
    note_length_s: float = 0.05,
    drum_map: dict[str, int] | None = None,
) -> list[tuple[int, float, float, int]]:
    """Convert canonical drum hits into (pitch, start, end, velocity) tuples.

    Drum samplers ignore note length — the sample plays to completion — but
    GM still requires note-off, so we use a short fixed duration.
    """
    mapping = drum_map or GM_DRUM_MAP
    out: list[tuple[int, float, float, int]] = []
    for h in hits:
        if h.label not in mapping:
            raise ValueError(f"No GM mapping for label {h.label!r}")
        pitch = mapping[h.label]
        out.append((pitch, h.time_s, h.time_s + note_length_s, h.velocity))
    return out
