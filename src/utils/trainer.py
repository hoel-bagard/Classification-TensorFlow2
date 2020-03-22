import time

import tensorflow as tf

from src.dataset.dataset_creator import DatasetCreator


class Trainer:
    def __init__(self, model, optimizer, loss_fn, dataset: DatasetCreator):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        # Get datasets
        self.train_dataset = dataset.get_train()
        self.val_dataset = dataset.get_val()

        # Define metrics
        self.train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
        self.train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="train_acc")
        self.val_loss_metric = tf.keras.metrics.Mean(name="val_loss")
        self.val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="train_acc")

        self.train_steps_per_epoch = dataset.train_dataset_size // dataset.batch_size
        self.val_steps_per_epoch = dataset.val_dataset_size // dataset.batch_size

    @tf.function
    def train_step(self, x, y_true):
        with tf.GradientTape() as tape:
            y_pred = self.model(x)
            loss = self.loss_fn(y_true, y_pred)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.train_loss_metric.update_state(loss)
        self.train_acc_metric.update_state(y_true, y_pred)

    def train_epoch(self):
        self.train_loss_metric.reset_states()
        self.train_acc_metric.reset_states()
        for step, (x_batch, y_batch) in enumerate(self.train_dataset):
            step_start_time = time.time()
            self.train_step(x_batch, y_batch)

            epoch_progress = int(30 * (step/self.train_steps_per_epoch))
            print(f"{step}/{self.train_steps_per_epoch} [" + epoch_progress*"=" + ">" +
                  (30-epoch_progress)*"." + f"] - loss: {self.train_loss_metric.result():.5f}  -  ",
                  f"Step time: {time.time() - step_start_time:.5f}s",
                  end='\r', flush=True)

        return self.train_loss_metric.result(), self.train_acc_metric.result()

    @tf.function
    def val_step(self, x, y_true):
        y_pred = self.model(x)
        loss = self.loss_fn(y_true, y_pred)
        self.val_loss_metric.update_state(loss)
        self.val_acc_metric.update_state(y_true, y_pred)

    def val_epoch(self):
        self.val_loss_metric.reset_states()
        self.val_acc_metric.reset_states()
        for step, (imgs_batch, labels_batch) in enumerate(self.val_dataset):
            step_start_time = time.time()

            self.val_step(imgs_batch, labels_batch)

            epoch_progress = int(30 * (step/self.val_steps_per_epoch))
            print(f"{step}/{self.val_steps_per_epoch} [" + epoch_progress*"=" + ">" +
                  (30-epoch_progress)*"." + f"] - loss: {self.val_loss_metric.result():.5f}  -  ",
                  f"Step time: {time.time() - step_start_time:.5f}s",
                  end='\r', flush=True)
        return self.val_loss_metric.result(), self.val_acc_metric.result()
