"""Step 4 - Train the LSTM on the preprocessed note sequences."""
import argparse
import json
import os

from tensorflow import keras

import config
from collect_data import collect
from model import build_model
from preprocess import build_dataset, build_vocab, load_all_tokens


def train(epochs: int = config.EPOCHS, batch_size: int = config.BATCH_SIZE):
    collect()                                   # makes sure MIDI data exists
    all_tokens = load_all_tokens()
    vocab = build_vocab(all_tokens)
    X, y = build_dataset(all_tokens, vocab)
    print(f"[train] {len(y['top_out'])} training sequences | melody tokens: {len(vocab['top'])} "
          f"| bass tokens: {len(vocab['bass'])} | duration tokens: {len(vocab['dur'])}")

    os.makedirs(os.path.dirname(config.MODEL_FILE), exist_ok=True)
    with open(config.VOCAB_FILE, "w", encoding="utf-8") as f:
        json.dump(vocab, f)

    model = build_model(len(vocab['top']), len(vocab['bass']), len(vocab['dur']))
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(config.MODEL_FILE, monitor="loss",
                                        save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="loss", patience=6,
                                      restore_best_weights=True),
    ]
    model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=callbacks)
    model.save(config.MODEL_FILE)
    print(f"[train] Model saved to {config.MODEL_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    args = parser.parse_args()
    train(args.epochs, args.batch)
