"""Convert generated tokens to a playable WAV file (pure numpy, no extra software)."""
import wave

import numpy as np

SAMPLE_RATE = 22050


def _tone(freq: float, seconds: float) -> np.ndarray:
    n = max(int(SAMPLE_RATE * seconds), 1)
    t = np.arange(n) / SAMPLE_RATE
    wave_ = (np.sin(2 * np.pi * freq * t)
             + 0.5 * np.sin(2 * np.pi * 2 * freq * t)
             + 0.25 * np.sin(2 * np.pi * 3 * freq * t))
    attack = min(int(0.01 * SAMPLE_RATE), n // 2)
    release = min(int(0.08 * SAMPLE_RATE), n // 2)
    env = np.ones(n)
    if attack:
        env[:attack] = np.linspace(0, 1, attack)
    if release:
        env[-release:] = np.linspace(1, 0, release)
    env *= np.exp(-1.5 * t / max(seconds, 0.05))      # gentle piano-like decay
    return wave_ * env


def tokens_to_wav(tokens, path: str, bpm: int = 90):
    """tokens look like '60_1.0' or '48.55.64_0.5' or 'R_0.5'."""
    sec_per_beat = 60.0 / bpm
    pieces = []
    for tok in tokens:
        pitches, dur = tok.rsplit("_", 1)
        seconds = float(dur) * sec_per_beat
        if pitches == "R":
            pieces.append(np.zeros(max(int(SAMPLE_RATE * seconds), 1)))
            continue
        midis = [int(p) for p in pitches.split(".")]
        mix = sum(_tone(440.0 * 2 ** ((m - 69) / 12), seconds) for m in midis)
        pieces.append(mix / max(len(midis), 1))
    audio = np.concatenate(pieces) if pieces else np.zeros(SAMPLE_RATE)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.8
    data = (audio * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(data.tobytes())
