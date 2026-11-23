"""Step 2 - Preprocess MIDI files into note sequences with music21.

Every musical event (a note, or a chord of several notes) becomes one text token
like  "60_1.0"  or  "48.55.64_0.5"  =  <midi pitches>_<duration in beats>.
"""
import glob
import os
import pickle

import numpy as np
from music21 import chord, converter, interval, key, note, pitch

import config

DURATIONS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]


def quantize_duration(quarter_length) -> float:
    """Snap a duration to the nearest value in DURATIONS."""
    ql = float(quarter_length)
    return min(DURATIONS, key=lambda d: abs(d - ql))


def transpose_to_c(score):
    """Transpose a score to C major / A minor so all songs share one key."""
    try:
        k = score.analyze("key")
        target = pitch.Pitch("C" if k.mode == "major" else "A")
        return score.transpose(interval.Interval(k.tonic, target))
    except Exception:
        return score                         # if analysis fails keep original


def parse_midi_file(path: str):
    """Return a list of tokens for one MIDI file."""
    score = converter.parse(path)
    score = transpose_to_c(score)           # one common key -> smaller vocabulary
    flat = score.chordify().flatten()      # merge all parts into chords/notes
    tokens = []
    for el in flat.notesAndRests:
        dur = quantize_duration(el.duration.quarterLength)
        if isinstance(el, chord.Chord):
            midis = sorted(p.midi for p in el.pitches)
            low, high = midis[0], midis[-1]          # bass note + melody note
            pitches = str(low) if low == high else f"{low}.{high}"
            tokens.append(f"{pitches}_{dur}")
        elif isinstance(el, note.Note):
            tokens.append(f"{el.pitch.midi}_{dur}")
        elif isinstance(el, note.Rest):
            tokens.append(f"R_{dur}")
    return tokens


def load_all_tokens(force: bool = False):
    """Parse every MIDI file in data/midi (cached in data/notes.pkl)."""
    if os.path.exists(config.NOTES_FILE) and not force:
        with open(config.NOTES_FILE, "rb") as f:
            return pickle.load(f)

    files = sorted(glob.glob(os.path.join(config.MIDI_DIR, "*.mid")) +
                   glob.glob(os.path.join(config.MIDI_DIR, "*.midi")))
    if not files:
        raise FileNotFoundError(
            "No MIDI files found in data/midi. Run  python collect_data.py  first.")

    all_tokens = []
    for i, path in enumerate(files, 1):
        try:
            toks = parse_midi_file(path)
            all_tokens.append(toks)
            print(f"[preprocess] {i}/{len(files)} {os.path.basename(path)} -> {len(toks)} events")
        except Exception as exc:
            print(f"[preprocess] skipped {os.path.basename(path)}: {exc}")

    all_tokens = [t for t in all_tokens if len(t) > config.SEQUENCE_LENGTH + 1]
    if not all_tokens:
        raise RuntimeError("No usable MIDI files (all were too short or unreadable).")

    os.makedirs(os.path.dirname(config.NOTES_FILE), exist_ok=True)
    with open(config.NOTES_FILE, "wb") as f:
        pickle.dump(all_tokens, f)
    return all_tokens


def split_token(token: str):
    """'48.72_0.5' -> ('72', '48', '0.5')  =  (melody, bass, duration)."""
    pitches, dur = token.rsplit("_", 1)
    if pitches == "R":
        return "R", "R", dur
    parts = pitches.split(".")
    return parts[-1], parts[0], dur


def _pitch_key(x: str):
    return (x != "R", int(x) if x != "R" else 0)


def build_vocab(all_tokens):
    parts = [split_token(t) for song in all_tokens for t in song]
    return {
        "top": sorted({a for a, _, _ in parts}, key=_pitch_key),
        "bass": sorted({b for _, b, _ in parts}, key=_pitch_key),
        "dur": sorted({c for _, _, c in parts}, key=float),
    }


def build_dataset(all_tokens, vocab, seq_len: int = config.SEQUENCE_LENGTH):
    """Inputs: last `seq_len` events.  Targets: next melody, bass and duration."""
    t2i = {t: i for i, t in enumerate(vocab["top"])}
    b2i = {t: i for i, t in enumerate(vocab["bass"])}
    d2i = {t: i for i, t in enumerate(vocab["dur"])}

    Xt, Xb, Xd, yt, yb, yd = [], [], [], [], [], []
    for song in all_tokens:                 # windows never cross song boundaries
        parts = [split_token(t) for t in song]
        ti = [t2i[a] for a, _, _ in parts]
        bi = [b2i[b] for _, b, _ in parts]
        di = [d2i[c] for _, _, c in parts]
        for i in range(len(ti) - seq_len):
            Xt.append(ti[i:i + seq_len])
            Xb.append(bi[i:i + seq_len])
            Xd.append(di[i:i + seq_len])
            yt.append(ti[i + seq_len])
            yb.append(bi[i + seq_len])
            yd.append(di[i + seq_len])
    arr = lambda a: np.array(a, dtype="int32")
    X = {"top_in": arr(Xt), "bass_in": arr(Xb), "dur_in": arr(Xd)}
    y = {"top_out": arr(yt), "bass_out": arr(yb), "dur_out": arr(yd)}
    return X, y
