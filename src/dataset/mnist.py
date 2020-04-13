import glob
import os

import cv2
import numpy as np
from PIL import Image

from config.model_config import ModelConfig


def load_mnist(data_path: str, mode: str):
    """
    Args:
        data_path: Path to the root folder of the dataset, it is expected to contain "Validation" and "Train".
        mode: Either "Train" or "Validation", depending on which dataset to load.
    """
    data_path = os.path.join(data_path, mode)
    imgs = []
    labels = []
    for i in range(10):
        for entry in glob.glob(os.path.join(data_path, str(i), "*.png"), recursive=True):
            print("Loading data {}".format(entry), end="\r")
            img = np.asarray(Image.open(entry))
            img = cv2.resize(img, (ModelConfig.IMG_SIZE, ModelConfig.IMG_SIZE), interpolation=cv2.INTER_AREA)
            imgs.append(img)
            labels.append(i)

    imgs = np.asarray(np.expand_dims(imgs, -1), dtype=np.float32)
    labels = np.asarray(labels, dtype=np.uint8)

    return imgs, labels
