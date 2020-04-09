import os
import glob

import cv2
import tensorflow as tf
import numpy as np
from PIL import Image

from config.model_config import ModelConfig


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
                imgs, labels = self._load_mnist(data_path, "Train")
            elif dataset_name == "Fashion-MNIST":
                imgs, labels = self._load_fashion_mnist(data_path, "Train")
            elif dataset_name == "CatVsDog":
                imgs, labels = self._load_cat_vs_dog(data_path, "Train")
            else:
                print("Wrong dataset name")
                return
        self.train_dataset = tf.data.Dataset.from_tensor_slices((imgs, labels))
        self.train_dataset = DatasetCreator._convert_image_dtype(self.train_dataset)  # Convert to float range
        self.train_dataset = self.train_dataset.shuffle(self.train_dataset_size, reshuffle_each_iteration=True)
        self.train_dataset = self.train_dataset.batch(self.batch_size)
        print('Train data loaded' + str(' ' * (os.get_terminal_size()[0] - 17)))

        # Prepare the training dataset.
        self.val_dataset_size: int
        if pickle:
            imgs, labels = self._load_pickle(data_path, dataset_name, "Validation")
        else:
            if dataset_name == "MNIST":
                imgs, labels = self._load_mnist(data_path, "Validation")
            elif dataset_name == "Fashion-MNIST":
                imgs, labels = self._load_fashion_mnist(data_path, "Validation")
            elif dataset_name == "CatVsDog":
                imgs, labels = self._load_cat_vs_dog(data_path, "Validation")
            else:
                print("Wrong dataset name")
                return
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

    def _load_mnist(self, data_path: str, mode: str):
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

        self.input_shape = imgs.shape[1:]
        if mode == "Train":
            self.train_dataset_size = len(labels)
        if mode == "Validation":
            self.val_dataset_size = len(labels)

        return imgs, labels

    def _load_cat_vs_dog(self, data_path: str, mode: str) -> (np.ndarray, np.ndarray):
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

        self.input_shape = imgs.shape[1:]
        if mode == "Train":
            self.train_dataset_size = len(labels)
        if mode == "Validation":
            self.val_dataset_size = len(labels)

        return imgs, labels

    def _load_fashion_mnist(self, data_path: str, mode: str) -> (np.ndarray, np.ndarray):
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

        self.input_shape = resized_imgs.shape[1:]
        if mode == "Train":
            self.train_dataset_size = nb_images
        if mode == "Validation":
            self.val_dataset_size = nb_images

        return resized_imgs, labels

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
                imgs, labels = self._load_mnist(data_path, mode)
            elif dataset_name == "CatVsDog":
                imgs, labels = self._load_cat_vs_dog(data_path, mode)
            else:
                print("Wrong dataset name")
                return -1
            np.save(imgs_pickle_path, imgs)
            np.save(labels_pickle_path, labels)

        return imgs, labels
