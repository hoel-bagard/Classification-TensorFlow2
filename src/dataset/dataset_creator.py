import os

import tensorflow as tf
import numpy as np

from src.dataset.mnist import load_mnist
from src.dataset.cat_vs_dog import load_cat_vs_dog
from src.dataset.fashion_mnist import load_fashion_mnist


class DatasetCreator:
    def __init__(self, data_path: str, dataset_name: str, batch_size: int, cache: bool = True, pickle: bool = True):
        """ Creates desired dataset.
        Args:
            dataset_name: Either MNIST or CatVsDog
        """
        self.batch_size: int = batch_size
        self.input_shape: int
        self.classes_nb: int
        data_path = os.path.join(data_path, dataset_name)

        if dataset_name == "MNIST" or dataset_name == "Fashion-MNIST":
            self.classes_nb = 10
        elif dataset_name == "CatVsDog":
            self.classes_nb = 2

        # Prepare the training dataset.
        self.train_dataset_size: int
        if pickle:
            imgs, labels = self._load_pickle(data_path, dataset_name, "Train")
        else:
            if dataset_name == "MNIST":
                imgs, labels = load_mnist(data_path, "Train")
            elif dataset_name == "Fashion-MNIST":
                imgs, labels = load_fashion_mnist(data_path, "Train")
            elif dataset_name == "CatVsDog":
                imgs, labels = load_cat_vs_dog(data_path, "Train")
            else:
                print("Wrong dataset name")
                return
        self.input_shape = imgs.shape[1:]
        self.train_dataset_size = len(labels)

        self.train_dataset = tf.data.Dataset.from_tensor_slices((imgs, labels))
        self.train_dataset = DatasetCreator._convert_image_dtype(self.train_dataset)  # Convert to float range
        self.train_dataset = self.train_dataset.shuffle(self.train_dataset_size, reshuffle_each_iteration=True)
        self.train_dataset = self.train_dataset.batch(self.batch_size)
        print('Train data loaded' + str(' ' * (os.get_terminal_size()[0] - 17)))

        # Prepare the validation dataset.
        self.val_dataset_size: int
        if pickle:
            imgs, labels = self._load_pickle(data_path, dataset_name, "Validation")
        else:
            if dataset_name == "MNIST":
                imgs, labels = load_mnist(data_path, "Validation")
            elif dataset_name == "Fashion-MNIST":
                imgs, labels = load_fashion_mnist(data_path, "Validation")
            elif dataset_name == "CatVsDog":
                imgs, labels = load_cat_vs_dog(data_path, "Validation")
            else:
                print("Wrong dataset name")
                return
        self.val_dataset_size = len(labels)

        self.val_dataset = tf.data.Dataset.from_tensor_slices((imgs, labels))
        self.val_dataset = DatasetCreator._convert_image_dtype(self.val_dataset)
        self.val_dataset = self.val_dataset.shuffle(self.val_dataset_size, reshuffle_each_iteration=True)
        self.val_dataset = self.val_dataset.batch(self.batch_size)
        print('Validation data loaded' + str(' ' * (os.get_terminal_size()[0] - 16)))

        if cache:
            self.train_dataset = self.train_dataset.cache()  # speed things up
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

    def _load_pickle(self, data_path: str, dataset_name: str, mode: str):
        imgs_pickle_path = os.path.join(data_path, mode, "imgs.npy")
        labels_pickle_path = os.path.join(data_path, mode, "labels.npy")
        if os.path.exists(imgs_pickle_path) and os.path.exists(imgs_pickle_path):
            print("Loading from pickle")
            imgs = np.load(imgs_pickle_path)
            labels = np.load(labels_pickle_path)
            if mode == "Train":
                self.train_dataset_size = len(labels)
            if mode == "Validation":
                self.val_dataset_size = len(labels)
            self.input_shape = imgs.shape[1:]
        else:
            if dataset_name == "MNIST":
                imgs, labels = load_mnist(data_path, mode)
            if dataset_name == "Fashion-MNIST":
                imgs, labels = load_fashion_mnist(data_path, mode)
            elif dataset_name == "CatVsDog":
                imgs, labels = load_cat_vs_dog(data_path, mode)
            else:
                print("Wrong dataset name")
                return -1
            np.save(imgs_pickle_path, imgs)
            np.save(labels_pickle_path, labels)

        return imgs, labels
