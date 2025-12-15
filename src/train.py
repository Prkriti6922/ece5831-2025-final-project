import os
import glob
import tensorflow as tf
from src.dataloader import build_dataset
from src.model import build_colorization_model

TRAIN_DIR = "data/train"
VAL_DIR = "data/test"  # optional small validation set

train_paths = glob.glob(TRAIN_DIR + "/*.jpg")
val_paths = glob.glob(VAL_DIR + "/*.jpg")

print("Training images:", len(train_paths))
print("Validation images:", len(val_paths))

train_ds = build_dataset(train_paths, batch_size=16, training=True)
val_ds = build_dataset(val_paths, batch_size=16, training=False)

model = build_colorization_model()
model.compile(optimizer="adam", loss="mse")

history = model.fit(train_ds, validation_data=val_ds, epochs=10)

os.makedirs("models", exist_ok=True)
model.save("models/colorization_model.h5")
print("Model saved to models/colorization_model.h5")
