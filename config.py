"""Central settings for the whole project."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MIDI_DIR = os.path.join(BASE_DIR, "data", "midi")
NOTES_FILE = os.path.join(BASE_DIR, "data", "notes.pkl")
MODEL_FILE = os.path.join(BASE_DIR, "models", "music_model.keras")
VOCAB_FILE = os.path.join(BASE_DIR, "models", "vocab.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

SEQUENCE_LENGTH = 32     # how many previous notes the model looks at
EPOCHS = 30
BATCH_SIZE = 64
LSTM_UNITS = 192
NUM_BACH_CHORALES = 120   # how many chorales to export as MIDI by collect_data.py
