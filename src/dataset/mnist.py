import os
import glob

import tensorflow as tf
import numpy as np
from PIL import Image


class MNISTDatasetCreator:
    def __init__(self, data_path: str, batch_size: int, cache: bool = True, pickle: bool = True):
        self.batch_size: int = batch_size
        self.input_shape: int

        # Prepare the training dataset.
        self.train_dataset_size: int
        if pickle:
            imgs, labels = self._load_pickle(data_path, "Train")
        else:
            imgs, labels = self._load_mnist(data_path, "Train")
        self.train_dataset = tf.data.Dataset.from_tensor_slices((imgs, labels))
        self.train_dataset = MNISTDatasetCreator._convert_image_dtype(self.train_dataset)  # Convert to float range
        self.train_dataset = self.train_dataset.shuffle(self.train_dataset_size, reshuffle_each_iteration=True)
        self.train_dataset = self.train_dataset.batch(self.batch_size)
        print('Train data loaded' + str(' ' * (os.get_terminal_size()[0] - 17)))

        # Prepare the training dataset.
        self.val_dataset_size: int
        if pickle:
            imgs, labels = self._load_pickle(data_path, "Test")
        else:
            imgs, labels = self._load_mnist(data_path, "Test")
        self.val_dataset = tf.data.Dataset.from_tensor_slices((imgs, labels))
        self.val_dataset = MNISTDatasetCreator._convert_image_dtype(self.val_dataset)
        self.val_dataset = self.val_dataset.shuffle(self.train_dataset_size, reshuffle_each_iteration=True)
        self.val_dataset = self.val_dataset.batch(self.batch_size)
        print('Validation data loaded' + str(' ' * (os.get_terminal_size()[0] - 16)))

        if cache:
            self.train_dataset = self.train_dataset.cache()  # speed things up considerably
            self.val_dataset = self.val_dataset.cache()

        print(f"Loaded {self.train_dataset_size} train data and {self.val_dataset_size} val data")

    def get_train(self):
        return self.train_dataset

    def get_val(self):
        return self.val_dataset

    @staticmethod
    def _convert_image_dtype(dataset):
        return dataset.map(
            lambda image, label: (
                tf.image.convert_image_dtype(image, tf.float32),
                label,
            )
        )

    def _load_mnist(self, data_path: str, mode: str):
        """
        Args:
            data_path: Path to the root folder of the dataset, this folder is expected to contain "Test" and "Train".
            mode: Either "Train" or "Test", depending on which dataset to load.
        """
        data_path = os.path.join(data_path, mode)
        imgs = []
        labels = []
        for i in range(10):
            for entry in glob.glob(os.path.join(data_path, str(i), "*.png"), recursive=True):
                print("Loading data {}".format(entry), end="\r")
                image = Image.open(entry)
                imgs.append(np.asarray(image))
                labels.append(i)

        imgs = np.asarray(np.expand_dims(imgs, -1), dtype=np.float32)
        labels = np.asarray(labels, dtype=np.uint8)

        self.input_shape = imgs.shape[1:]
        if mode == "Train":
            self.train_dataset_size = len(labels)
        if mode == "Test":
            self.val_dataset_size = len(labels)

        return imgs, labels

    def _load_pickle(self, data_path: str, mode: str):
        imgs_pickle_path = os.path.join(data_path, mode, "imgs.npy")
        labels_pickle_path = os.path.join(data_path, mode, "labels.npy")
        if os.path.exists(imgs_pickle_path) and os.path.exists(imgs_pickle_path):
            imgs = np.load(imgs_pickle_path)
            labels = np.load(labels_pickle_path)
            if mode == "Train":
                self.train_dataset_size = len(labels)
            if mode == "Test":
                self.val_dataset_size = len(labels)
            self.input_shape = imgs.shape[1:]
        else:
            imgs, labels = self._load_mnist(data_path, mode)
            np.save(imgs_pickle_path, imgs)
            np.save(labels_pickle_path, labels)

        return imgs, labels
