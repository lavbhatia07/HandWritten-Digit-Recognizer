"""
Train a deployment-safe handwritten digit model for digits 1 through 9.

The model is saved in the modern Keras format as digit_model.keras so Render
does not need to deserialize the older, incompatible digit_model.h5 file.
"""

import os
from pathlib import Path

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", BASE_DIR / "digit_model.keras")).resolve()
NUM_CLASSES = 9


def build_model() -> Sequential:
    return Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.35),
        Dense(NUM_CLASSES, activation="softmax"),
    ])


def prepare_digits_1_to_9(images: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keep = labels != 0
    images = images[keep].astype("float32") / 255.0
    labels = labels[keep] - 1

    images = images.reshape(-1, 28, 28, 1)
    labels = to_categorical(labels, NUM_CLASSES)
    return images, labels


def main() -> None:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = prepare_digits_1_to_9(x_train, y_train)
    x_test, y_test = prepare_digits_1_to_9(x_test, y_test)

    model = build_model()
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=2,
        restore_best_weights=True,
    )

    model.fit(
        x_train,
        y_train,
        batch_size=128,
        epochs=6,
        validation_data=(x_test, y_test),
        callbacks=[early_stop],
        verbose=1,
    )

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy for digits 1-9: {accuracy * 100:.2f}%")

    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
