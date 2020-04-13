import glob
import os

import cv2
import numpy as np
from PIL import Image

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
    for entry in glob.glob(os.path.join(data_path, "*.jpg")):
        print("Loading data {}   ".format(entry), end="\r")
        img = np.asarray(Image.open(entry))
        if ModelConfig.IMG_SIZE:
            img = cv2.resize(img, (ModelConfig.IMG_SIZE, ModelConfig.IMG_SIZE), interpolation=cv2.INTER_AREA)
        imgs.append(img)
        file_name = os.path.basename(entry)
        if file_name[:3] == "cat":
            labels.append(0)
        elif file_name[:3] == "dog":
            labels.append(1)
        else:
            print(f"Wrong file name: {file_name}")

    imgs = np.asarray(imgs, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.uint8)

    return imgs, labels
