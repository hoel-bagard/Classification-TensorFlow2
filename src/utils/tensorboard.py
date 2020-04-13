# import os
# from datetime import datetime

import tensorflow as tf
import numpy as np

from src.utils.draw import draw_pred


class TensorBoard():
    def __init__(self, tb_dir: str, max_outputs: int = 4):
        """
        Args:
            model: Model used, it is used to create the TensorBoard graph
            input_shape: Shape of the inputs given to the model
            max_outputs: Number of images kept and dislpayed in TensorBoard
        """
        super(TensorBoard, self).__init__()
        self.max_outputs = max_outputs

        # tb_file_path = os.path.join(tb_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.file_writer = tf.summary.create_file_writer(tb_dir)

    def write_metrics(self, loss: float, acc: float, epoch: int, mode="Train"):
        """
        Args:
            loss: loss value to be added to tensorboard
            epoch: Current epoch
            mode: Either "Train" or "Validation"
        """
        with self.file_writer.as_default():
            tf.summary.scalar("Loss/"+mode, loss, step=epoch)
            tf.summary.scalar("Accuracy/"+mode, acc, step=epoch)
            self.file_writer.flush()

    def write_lr(self, lr: float, epoch: int):
        """
        Args:
            lr: learning rate value to be added to tensorboard
            epoch: Current epoch
        """
        with self.file_writer.as_default():
            tf.summary.scalar("Learning Rate", lr, step=epoch)
            self.file_writer.flush()

    def write_predictions(self, imgs: np.ndarray, predictions: np.ndarray, labels: np.ndarray,
                          epoch: int, mode="Train"):
        """
        Args:
            imgs: images that have been fed to the network
            predictions: Model's predictions
            epoch: Current epoch
            mode: Either "Train" or "Validation"
        """
        imgs = np.asarray(imgs[:self.max_outputs], dtype=np.uint8)
        predictions = predictions[:self.max_outputs]
        labels = labels[:self.max_outputs]

        new_imgs = draw_pred(imgs, predictions, labels)
        # MNIST is black and white
        if new_imgs.shape[-1] != 3:
            new_imgs = np.expand_dims(new_imgs, -1)
        with self.file_writer.as_default():
            tf.summary.image(mode, new_imgs, max_outputs=self.max_outputs, step=epoch)
            self.file_writer.flush()
