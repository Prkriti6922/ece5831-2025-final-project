import tensorflow as tf


def compute_metrics(true_rgb, pred_rgb):
    true_rgb = tf.cast(true_rgb * 255, tf.uint8)
    pred_rgb = tf.cast(tf.clip_by_value(pred_rgb, 0, 1) * 255, tf.uint8)

    psnr = tf.image.psnr(true_rgb, pred_rgb, max_val=255).numpy()
    ssim = tf.image.ssim(true_rgb, pred_rgb, max_val=255).numpy()

    return float(psnr), float(ssim)
