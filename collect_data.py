"""Step 1 - Collect MIDI music data.

Exports Bach chorales (classical music) that ship inside the music21 library
into .mid files in data/midi/.  No internet connection is needed.
You can also copy your own .mid files (classical, jazz, ...) into data/midi/.
"""
import glob
import os

from music21 import corpus

import config


def collect(limit: int = config.NUM_BACH_CHORALES) -> int:
    os.makedirs(config.MIDI_DIR, exist_ok=True)
    existing = glob.glob(os.path.join(config.MIDI_DIR, "*.mid"))
    existing += glob.glob(os.path.join(config.MIDI_DIR, "*.midi"))
    if existing:
        print(f"[collect] {len(existing)} MIDI files already in data/midi - skipping export.")
        return len(existing)

    print(f"[collect] Exporting up to {limit} Bach chorales to MIDI ...")
    saved = 0
    for score in corpus.chorales.Iterator(1, limit, returnType="stream"):
        try:
            path = os.path.join(config.MIDI_DIR, f"bach_chorale_{saved + 1:03d}.mid")
            score.write("midi", fp=path)
            saved += 1
        except Exception as exc:  # skip any problematic file
            print(f"[collect] skipped one chorale: {exc}")
    print(f"[collect] Saved {saved} MIDI files in {config.MIDI_DIR}")
    return saved


if __name__ == "__main__":
    collect()
