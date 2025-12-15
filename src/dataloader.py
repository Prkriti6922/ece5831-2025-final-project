# src/dataloader.py
import tensorflow as tf
import tensorflow_io as tfio

IMG_SIZE = 128


def preprocess_image(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0

    lab = tfio.experimental.color.rgb_to_lab(img)

    L = lab[:, :, :1] / 100.0  # (H, W, 1)
    ab = lab[:, :, 1:] / 128.0  # (H, W, 2)

    return L, ab


def build_dataset(paths, batch_size=16, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices(paths)

    if shuffle:
        ds = ds.shuffle(1000)

    ds = ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds
