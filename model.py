"""Step 3 - LSTM model.

Inputs : last N melody notes, bass notes and durations
Outputs: next melody note, next bass note, next duration (3 softmax heads)
"""
from tensorflow import keras
from tensorflow.keras import layers

import config


def build_model(n_top: int, n_bass: int, n_dur: int, seq_len: int = config.SEQUENCE_LENGTH):
    top_in = keras.Input(shape=(seq_len,), dtype="int32", name="top_in")
    bass_in = keras.Input(shape=(seq_len,), dtype="int32", name="bass_in")
    dur_in = keras.Input(shape=(seq_len,), dtype="int32", name="dur_in")

    x = layers.Concatenate()([
        layers.Embedding(n_top, 32)(top_in),
        layers.Embedding(n_bass, 32)(bass_in),
        layers.Embedding(n_dur, 8)(dur_in),
    ])
    x = layers.LSTM(config.LSTM_UNITS, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(config.LSTM_UNITS)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)

    top_out = layers.Dense(n_top, activation="softmax", name="top_out")(x)
    bass_out = layers.Dense(n_bass, activation="softmax", name="bass_out")(x)
    dur_out = layers.Dense(n_dur, activation="softmax", name="dur_out")(x)

    model = keras.Model([top_in, bass_in, dur_in], [top_out, bass_out, dur_out])
    loss = "sparse_categorical_crossentropy"
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss={"top_out": loss, "bass_out": loss, "dur_out": loss},
        metrics={"top_out": "accuracy", "bass_out": "accuracy", "dur_out": "accuracy"},
    )
    return model
