import os

import cv2
import numpy as np

from config.model_config import ModelConfig


def load_fashion_mnist(data_path: str, mode: str) -> (np.ndarray, np.ndarray):
    """
    Args:
        data_path: Path to a folder containing the byte files.
        mode: Either "Train" or "Validation", depending on which dataset to load.
    """
    if mode == "Train":
        img_data_path = os.path.join(data_path, "train-images-idx3-ubyte")
        labels_data_path = os.path.join(data_path, "train-labels-idx1-ubyte")
    elif mode == "Validation":
        img_data_path = os.path.join(data_path, "t10k-images-idx3-ubyte")
        labels_data_path = os.path.join(data_path, "t10k-labels-idx1-ubyte")

    f = open(img_data_path, "rb")
    dt = np.dtype(np.uint32).newbyteorder('>')

    # Loading images
    f.read(4)  # magic_number, I don't use it
    nb_images: int = np.frombuffer(f.read(4), dtype=dt)[0]
    nb_rows: int = np.frombuffer(f.read(4), dtype=dt)[0]
    nb_cols: int = np.frombuffer(f.read(4), dtype=dt)[0]

    data = np.frombuffer(f.read(nb_rows * nb_cols * nb_images), dtype=np.uint8)
    imgs: np.ndarray = data.reshape(nb_images, nb_rows, nb_cols)
    resized_imgs: np.ndarray = np.empty((nb_images, ModelConfig.IMG_SIZE, ModelConfig.IMG_SIZE))
    for i in range(len(imgs)):
        resized_imgs[i] = cv2.resize(imgs[i],
                                     (ModelConfig.IMG_SIZE, ModelConfig.IMG_SIZE),
                                     interpolation=cv2.INTER_AREA)
    resized_imgs = np.asarray(np.expand_dims(imgs, -1), dtype=np.float32)

    # Loading the labels
    f = open(labels_data_path, "rb")
    f.read(4)  # magic_number, I don't use it
    nb_labels: int = np.frombuffer(f.read(4), dtype=dt)[0]

    labels: np.ndarray = np.frombuffer(f.read(nb_labels), dtype=np.uint8)

    return resized_imgs, labels
