# src/model.py
import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = 128


def build_colorization_model():
    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

    # Encoder (same as Colab)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(inp)
    x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(256, 3, strides=2, padding="same", activation="relu")(x)

    # Decoder
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)

    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)

    x = layers.UpSampling2D()(x)
    out = layers.Conv2D(2, 3, activation="tanh", padding="same")(x)

    return models.Model(inp, out)
