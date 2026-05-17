# quick-stim

A local CLI pipeline that takes a Spotify playlist URL and produces, for each track,
a per-song folder containing:

- `original.mp3` — the source audio (downloaded via spotdl + YouTube matching)
- `stems/{vocals,drums,bass,other}.wav` — 4-stem separation via demucs `htdemucs_ft`
- `midi/{vocals,bass,other,drums}.mid` — per-stem MIDI transcriptions
- `metadata.json` — tempo, key, time signature, duration, beats, processing log

Built as a starting point for cover arrangements in Logic Pro.

## Install

Targets macOS Apple Silicon (M-series). Linux x86 works too, modulo PyTorch
device differences (CUDA / CPU instead of MPS).

```bash
# system dep
brew install ffmpeg

# python deps
uv sync

# primary drum engine
uv sync --extra adtof

# (Magenta-Onsets-and-Frames is the documented fallback per spec, but it
# pins numpy==1.21.6 and CANNOT co-install with librosa/demucs. If you
# need it, set up a SEPARATE venv just for magenta drum inference; the
# pipeline's _try_magenta hook lazy-imports it at runtime.)
```

ADTOF model weights: `crnn-all` downloads on first use. If it fails to
download automatically, see https://github.com/MZehren/ADTOF for the
release artifact location.

## Run

```bash
python -m pipeline 'https://open.spotify.com/playlist/1wZfLEUcgEsDdKinBdO7Q3' \
  --output ./output \
  --songs-limit 1
```

Useful flags:

- `--songs-limit N` — process only first N tracks (validate end-to-end first).
- `--steps download,separate,transcribe,analyze` — run a subset.
- `--skip-melodic` / `--skip-drums` — iterate on one transcription type at a time.
- `--force` — overwrite existing outputs (default: skip if present).

Each step is idempotent: re-running skips work that's already done.

## Output layout

```
output/
  Arctic Monkeys - Do I Wanna Know/
    original.mp3
    stems/
      vocals.wav
      drums.wav
      bass.wav
      other.wav
    midi/
      vocals.mid
      bass.mid
      other.mid
      drums.mid
    metadata.json
```

## Expected runtime (per song, M-series Mac, ~3.5min track)

| step               | rough time         | notes                                  |
|--------------------|--------------------|----------------------------------------|
| spotdl download    | 10-30s             | network + ffmpeg re-encode             |
| demucs htdemucs_ft | 60-120s            | MPS-accelerated; sequential per song   |
| basic-pitch x3     | 30-60s parallel    | bass/other/vocals run concurrently     |
| ADTOF drums        | 20-40s             | CPU bound for the most part            |
| librosa analyze    | 5-10s              |                                        |
| **total**          | **~3-5 min/song**  | first run; re-runs skip cached outputs |

## Architecture

- `pipeline/storage.py` defines a minimal `Storage` protocol
  (`read_bytes`, `write_bytes`, `exists`, `list`, plus a `local_path` escape
  hatch for tools that demand filesystem paths). `LocalStorage` is the
  default impl. Every pipeline step takes a `Storage` handle — no raw
  `Path`/`open()` use in step code — so dropping in an `R2Storage` later
  doesn't require touching `download.py` / `separate.py` / etc.
- Each step is a single function with the shape
  `step(input_key: str, output_prefix_or_key: str, storage: Storage, ...) -> dict`.
  No implicit "current song" state. Suitable for splitting into Modal
  functions or container tasks later.
- `pipeline/runner.py` is the only module that knows the whole sequence.

## Drum transcription

The drum stem MIDI uses GM percussion mapping defined in
`pipeline/drum_midi_mapping.py`:

| ADTOF class | GM note    | name               |
|-------------|------------|--------------------|
| kick        | 36 (C1)    | Acoustic Bass Drum |
| snare       | 38 (D1)    | Acoustic Snare     |
| hihat       | 42 (F#1)   | Closed Hi-Hat      |
| tom         | 45 (A1)    | Low Tom            |
| cymbal      | 49 (C#2)   | Crash Cymbal 1     |

Edit `GM_DRUM_MAP` in that file to adapt to a different drum sampler
(SSD5, Superior Drummer, BFD all use the same GM core but vary on
articulations and velocity layers).

### Engine fallback chain

The pipeline tries drum engines in this order, recording which one fired
in `metadata.json` (`drum_engine` field):

1. **ADTOF-pytorch CRNN** (primary) — best accuracy, 5-class output.
2. **Magenta Onsets-and-Frames drums** (documented fallback) — installed
   via `uv sync --extra magenta`. Note: the OAF drum inference loader is
   not fully wired up in this POC; if you need to use it, the stub in
   `transcribe_drums.py:_try_magenta` is the integration point.
3. **librosa band-onset detection** (last resort) — splits the drum stem
   into kick / snare / hi-hat frequency bands and runs `onset_detect` on
   each. Quality is intentionally limited and flagged clearly in the
   output metadata. Use this only to sanity-check the rest of the pipeline
   when neither ADTOF nor Magenta is installed.

## Limitations (honest)

- **Drums-to-MIDI**: ~80-90% accuracy on isolated drum stems with ADTOF.
  Lower for cymbal-heavy passages (rides vs. crashes vs. chinas all
  collapse to one "cymbal" class, mapped to crash by default). Plan to
  hand-clean cymbal hits in Logic.
- **Bass-to-MIDI**: high accuracy. Bass is monophonic enough that
  basic-pitch does well out of the box. Watch for octave errors on
  very low notes (below E1).
- **Other-to-MIDI (guitars/keys/synths)**: 60-70% accuracy. Treat as a
  scaffold — chord shapes and rhythm are usable, exact voicings will need
  cleanup. If you hit a wall, see "Upgrade paths" below.
- **Vocals-to-MIDI**: reference only. basic-pitch with the lowered
  thresholds catches phrasing and pitch contour but note boundaries on
  vibrato / melisma are unreliable. Useful for: "what was that melody
  again?". Not useful for: a melody you'd quantize and re-perform without
  edits.
- **Time signature detection**: heuristic, defaults to 4/4 unless beats
  are very irregular. Verify manually for any 3/4, 6/8, 5/4, etc. songs.
- **Key detection**: Krumhansl-Schmuckler chroma correlation. Works well
  for clearly diatonic pop; less reliable for modal, key-changing, or
  chromatic material.

## Upgrade paths

- If the `other.mid` stem is unusable for the songs you cover, swap in a
  hosted polyphonic transcription API:
  - [Samplab](https://samplab.com) — polyphonic, has API
  - [Songscription](https://songscription.com) — full-mix transcription
- If 4-stem separation isn't granular enough (you want crash/ride
  separated, or piano vs guitar in "other"), try the Jarredou 6-stem
  demucs model: it splits "other" into guitar/piano and "drums" into
  kick/snare/cymbals. Drop-in: swap `htdemucs_ft` for the 6s model in
  `pipeline/separate.py`.
- If MPS is too slow on long tracks, run demucs with `--two-stems vocals`
  for a vocals/instrumental split as a cheap preview.

## Constraints honored

- No cloud APIs — all processing local.
- Logic Pro is never invoked (its stem splitter and audio-to-MIDI have
  no CLI; this is a hard limit).
- spotdl YouTube-matching failures are logged to `metadata.json`
  (`download_error` field) and the pipeline continues with the next song.
