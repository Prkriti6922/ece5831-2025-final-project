# src/infer.py
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

IMG_SIZE = 128


def load_L(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype("float32")
    L = lab[:, :, :1] / 100.0  # normalize exactly like training

    return L


def predict_color(model, L):
    pred_ab = model.predict(L[None, ...])[0] * 128.0

    lab = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype="float32")

    # OpenCV LAB expects:
    # L in [0,255], a,b in [0,255] with 128 = zero
    lab[:, :, 0] = L[:, :, 0] * 255.0
    lab[:, :, 1] = pred_ab[:, :, 0] + 128.0
    lab[:, :, 2] = pred_ab[:, :, 1] + 128.0

    rgb = cv2.cvtColor(lab.astype("uint8"), cv2.COLOR_LAB2RGB)
    rgb = np.clip(rgb / 255.0, 0, 1)

    return rgb


def infer(image_path, model_path="models/colorization_model.h5", save=True):
    os.makedirs("results", exist_ok=True)

    model = load_model(model_path, compile=False)

    L = load_L(image_path)
    rgb = predict_color(model, L)

    # Visualization
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    plt.title("Grayscale (L)")
    plt.imshow(L[:, :, 0], cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Colorized")
    plt.imshow(rgb)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Original")
    orig = cv2.imread(image_path)
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    orig = cv2.resize(orig, (IMG_SIZE, IMG_SIZE))
    plt.imshow(orig)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    if save:
        out_path = os.path.join("results", os.path.basename(image_path))
        plt.imsave(out_path, rgb)
        print(f"Saved colorized image to {out_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python src/infer.py <image_path> [model_path]")
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "models/colorization_model.h5"

    infer(image_path, model_path=model_path)
