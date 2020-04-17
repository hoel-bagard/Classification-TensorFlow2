import random

import tensorflow as tf

from config.model_config import ModelConfig


def color_augment(x: tf.Tensor, y: tf.Tensor):
    """Color augmentation
    Args:
        x: Images
        y: Labels
    """
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.1)
    x = tf.image.random_contrast(x, 0.6, 1.4)
    x = tf.minimum(x, 1.0)
    x = tf.maximum(x, 0.0)
    return x, y


def rot_flip_augment(x: tf.Tensor, y: tf.Tensor):
    """Random 90 degrees rotations and flips augmentation
    Args:
        x: Images
        y: Labels
    """
    # Rotation
    x = tf.image.rot90(x, k=random.randint(0, 3))
    # Zoom
    x = tf.image.random_crop(x, (ModelConfig.IMG_SIZE, ModelConfig.IMG_SIZE, 3))
    x = tf.image.resize(x, (ModelConfig.IMG_SIZE, ModelConfig.IMG_SIZE))
    # Flip
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    return x, y
