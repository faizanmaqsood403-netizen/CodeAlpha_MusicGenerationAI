"""Step 5 - Generate new music and save it as MIDI + WAV (and optionally play it)."""
import argparse
import json
import os
import sys
import time

import numpy as np
from music21 import chord, instrument, note, stream, tempo
from tensorflow import keras

import config
from audio import tokens_to_wav
from preprocess import load_all_tokens, split_token


def sample(probs: np.ndarray, temperature: float) -> int:
    probs = np.asarray(probs, dtype="float64")
    probs = np.log(probs + 1e-9) / max(temperature, 1e-3)
    probs = np.exp(probs - probs.max())
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


def generate_tokens(model, vocab, seeds, length: int, temperature: float):
    wt, wb, wd = (list(s) for s in seeds)
    out = []
    for _ in range(length):
        pt, pb, pd = model.predict(
            {"top_in": np.array([wt], dtype="int32"),
             "bass_in": np.array([wb], dtype="int32"),
             "dur_in": np.array([wd], dtype="int32")}, verbose=0)
        it, ib, idr = (sample(pt[0], temperature), sample(pb[0], temperature),
                       sample(pd[0], temperature))
        top, bass, dur = vocab["top"][it], vocab["bass"][ib], vocab["dur"][idr]
        if top == "R" or bass == "R":
            pitches = "R"
        elif top == bass:
            pitches = top
        else:
            low, high = sorted((int(top), int(bass)))
            pitches = f"{low}.{high}"
        out.append(f"{pitches}_{dur}")
        wt, wb, wd = wt[1:] + [it], wb[1:] + [ib], wd[1:] + [idr]
    return out


def tokens_to_midi(tokens, path: str, bpm: int = 90):
    part = stream.Part()
    part.insert(0, instrument.Piano())
    part.insert(0, tempo.MetronomeMark(number=bpm))
    offset = 0.0
    for tok in tokens:
        pitches, dur = tok.rsplit("_", 1)
        dur = float(dur)
        if pitches == "R":
            el = note.Rest(quarterLength=dur)
        elif "." in pitches:
            el = chord.Chord([int(p) for p in pitches.split(".")], quarterLength=dur)
        else:
            el = note.Note(int(pitches), quarterLength=dur)
        part.insert(offset, el)
        offset += dur
    part.write("midi", fp=path)


def open_file(path: str):
    """Play the file with the system default player."""
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)                      # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            os.system(f'open "{path}"')
        else:
            os.system(f'xdg-open "{path}" >/dev/null 2>&1')
    except Exception as exc:
        print(f"[generate] Could not auto-play ({exc}). Open the file manually: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=150, help="number of notes/chords")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="lower = safer, higher = more random")
    parser.add_argument("--bpm", type=int, default=90)
    parser.add_argument("--play", action="store_true", help="play the WAV when finished")
    args = parser.parse_args()

    if not os.path.exists(config.MODEL_FILE):
        raise SystemExit("Model not found. Run  python train.py  first.")

    model = keras.models.load_model(config.MODEL_FILE)
    with open(config.VOCAB_FILE, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    maps = {k: {v: i for i, v in enumerate(vocab[k])} for k in ("top", "bass", "dur")}

    # Random seed taken from the training music
    songs = [s for s in load_all_tokens() if len(s) > config.SEQUENCE_LENGTH]
    song = songs[np.random.randint(len(songs))]
    start = np.random.randint(0, len(song) - config.SEQUENCE_LENGTH)
    window = [split_token(t) for t in song[start:start + config.SEQUENCE_LENGTH]]
    seeds = ([maps["top"][a] for a, _, _ in window],
             [maps["bass"][b] for _, b, _ in window],
             [maps["dur"][c] for _, _, c in window])

    print(f"[generate] Composing {args.length} events ...")
    tokens = generate_tokens(model, vocab, seeds, args.length, args.temperature)

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    midi_path = os.path.join(config.OUTPUT_DIR, f"generated_{stamp}.mid")
    wav_path = os.path.join(config.OUTPUT_DIR, f"generated_{stamp}.wav")
    tokens_to_midi(tokens, midi_path, args.bpm)
    tokens_to_wav(tokens, wav_path, args.bpm)
    print(f"[generate] MIDI saved : {midi_path}")
    print(f"[generate] Audio saved: {wav_path}")

    if args.play:
        open_file(wav_path)


if __name__ == "__main__":
    main()
