import os
from datetime import datetime

import tensorflow as tf


class TensorBoard():
    def __init__(self, model: tf.keras.Model, input_shape: tuple, max_outputs=4, tb_dir="../logs/"):
        """
        Args:
            model: Model used, it is used to create the TensorBoard graph
            input_shape: Shape of the inputs given to the model
            max_outputs: Number of images kept and dislpayed in TensorBoard
        """
        super(TensorBoard, self).__init__()
        self.max_outputs = max_outputs

        tb_file_path = os.path.join(tb_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.file_writer = tf.summary.create_file_writer(tb_file_path)

        # Creates the TensorBoard graph
        # Result is not great, but it's a start. Ideally it should trace a whole training step
        @tf.function
        def trace_me(x):
            return model(x)
        tf.summary.trace_on(graph=True, profiler=False)
        trace_me(tf.expand_dims(tf.zeros(input_shape), 0))
        with self.file_writer.as_default():
            tf.summary.trace_export(name="OMG2.2", step=0)

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

    def write_predictions(self, imgs, predictions, epoch: int, mode="Train"):
        """
        Args:
            imgs: images that have been fed to the network
            predictions: Model's predictions
            epoch: Current epoch
            mode: Either "Train" or "Validation"
        """
        with self.file_writer.as_default():
            tf.summary.image(mode, imgs, max_outputs=self.max_outputs, step=epoch)
            self.file_writer.flush()
