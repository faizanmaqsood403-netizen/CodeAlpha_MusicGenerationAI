# CodeAlpha_Music_Generation

AI internship — **Task 3: Music Generation with AI** (CodeAlpha)

An LSTM neural network that learns patterns from classical music (Bach chorales, MIDI format)
and composes brand-new music, saved as a **MIDI** file and a playable **WAV** audio file.

## How it works (maps to the task requirements)
| Task requirement | Where |
|---|---|
| Collect MIDI music data (classical, jazz, ...) | `collect_data.py` exports Bach chorales to `data/midi/*.mid`. You can also drop your own `.mid` files there. |
| Preprocess into note sequences using `music21` | `preprocess.py` parses MIDI, transposes every song to C major / A minor, converts it to a sequence of events (melody note + bass note + duration) and builds sliding windows of 32 events. |
| Deep learning model with LSTM | `model.py` - Embeddings -> 2x LSTM -> Dense -> 3 outputs (next melody note, next bass note, next duration). |
| Train the model | `train.py` |
| Generate new sequences, convert to MIDI, play/save audio | `generate.py` writes `output/generated_*.mid` and `output/generated_*.wav` (`audio.py` is a built-in synthesizer, no extra software needed). `--play` opens the audio. |

## Project structure
```
CodeAlpha_Music_Generation/
├── config.py          # all settings (epochs, sequence length, paths)
├── collect_data.py    # step 1: get MIDI data
├── preprocess.py      # step 2: music21 preprocessing
├── model.py           # step 3: LSTM model
├── train.py           # step 4: training
├── generate.py        # step 5: generate MIDI + WAV
├── audio.py           # MIDI/notes -> WAV synthesizer
├── run_all.py         # runs everything in one command
├── requirements.txt
├── data/midi/         # training MIDI files
├── models/            # trained model (created by train.py)
└── output/            # generated music
```

## Setup
Python 3.9 - 3.12 recommended.
```bash
pip install -r requirements.txt
```

## Run
```bash
python run_all.py            # collect -> train -> generate -> play (everything)
```
or step by step:
```bash
python collect_data.py                 # 1. get MIDI data
python train.py                        # 2. train (default 30 epochs)
python generate.py --play              # 3. compose and play new music
```

Useful options:
```bash
python train.py --epochs 10                    # quicker (lower quality)
python generate.py --length 300                # longer piece
python generate.py --temperature 1.0           # more random / creative (default 0.8)
python generate.py --bpm 110                   # faster tempo
```
Every call of `generate.py` creates a different composition.

## Use your own music
Copy any `.mid` files (classical, jazz, ...) into `data/midi/`, delete `data/notes.pkl`, then run `python train.py`.

## Tech
Python, TensorFlow/Keras (LSTM), music21, NumPy
