import glob
import os

import cv2
import numpy as np

from config.model_config import ModelConfig


def load_cat_vs_dog(data_path: str, mode: str) -> (np.ndarray, np.ndarray):
    """
    Args:
        data_path: Path to the root folder of the dataset, it is expected to contain "Validation" and "Train".
        mode: Either "Train" or "Validation", depending on which dataset to load.
    """
    data_path = os.path.join(data_path, mode)
    imgs = []
    labels = []
    for image_path in glob.glob(os.path.join(data_path, "*.jpg")):
        print("Loading data {}   ".format(image_path), end="\r")
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if ModelConfig.IMG_SIZE:
            img = cv2.resize(img, (ModelConfig.IMG_SIZE, ModelConfig.IMG_SIZE), interpolation=cv2.INTER_AREA)
        imgs.append(img)

        file_name = os.path.basename(image_path)
        labels.append(0 if "cat" in file_name else 1)

    imgs = np.asarray(imgs, dtype=np.uint8)
    labels = np.asarray(labels, dtype=np.uint8)

    return imgs, labels
