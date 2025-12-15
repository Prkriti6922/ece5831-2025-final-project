import os
import shutil
import glob

SOURCE = "data/train2017"
TARGET = "data/train_small"

os.makedirs(TARGET, exist_ok=True)

all_images = glob.glob(SOURCE + "/*.jpg")[:5000]

for img in all_images:
    shutil.copy(img, TARGET)

print("Created dataset with", len(all_images), "images")
