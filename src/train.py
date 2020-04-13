import time
import os

import tensorflow as tf

from src.dataset.dataset_creator import DatasetCreator
from config.data_config import DataConfig
from config.model_config import ModelConfig
from src.utils.tensorboard import TensorBoard
from src.utils.trainer import Trainer


def train(model: tf.keras.Model, dataset: DatasetCreator):
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=ModelConfig.LR)
    loss_fn = tf.losses.SparseCategoricalCrossentropy(from_logits=False)
    trainer = Trainer(model, optimizer, loss_fn, dataset)

    if DataConfig.USE_TB:
        tensorboard = TensorBoard(tb_dir=DataConfig.TB_DIR)
        # Creates the TensorBoard Graph. The result is not great, but it's a start.
        tf.summary.trace_on(graph=True, profiler=False)
        trainer.val_step(tf.expand_dims(tf.zeros(dataset.input_shape), 0), 0)
        with tensorboard.file_writer.as_default():
            tf.summary.trace_export(name=ModelConfig.NETWORK_NAME, step=0)

    best_loss = 1000
    last_checkpoint_epoch = 0

    for epoch in range(ModelConfig.MAX_EPOCHS):
        print(f"\nEpoch {epoch}/{ModelConfig.MAX_EPOCHS}")
        epoch_start_time = time.time()

        train_loss, train_acc = trainer.train_epoch()

        if DataConfig.USE_TB:
            tensorboard.write_metrics(train_loss, train_acc, epoch)
            tensorboard.write_lr(ModelConfig.LR, epoch)

        if (train_loss < best_loss and DataConfig.USE_CHECKPOINT and
                epoch >= DataConfig.RECORD_DELAY and (epoch - last_checkpoint_epoch) >= DataConfig.CHECKPT_SAVE_FREQ):

            save_path = os.path.join(DataConfig.CHECKPOINT_DIR, f'train_{epoch}')
            print(f"Loss improved from {best_loss} to {train_loss}, saving model to {save_path}")
            best_loss = train_loss
            last_checkpoint_epoch = epoch
            model.save(save_path)

        print(f"\nEpoch loss: {train_loss}, Train accuracy: {train_acc}  -  Took {time.time() - epoch_start_time:.5f}s")

        # Validation and (expensive to compute) metrics
        if epoch % DataConfig.VAL_FREQ == 0 and epoch > DataConfig.RECORD_DELAY:
            validation_start_time = time.time()
            val_loss, val_acc = trainer.val_epoch()

            if DataConfig.USE_TB:
                tensorboard.write_metrics(val_loss, val_acc, epoch, mode="Validation")

                # Metrics for the Train dataset
                imgs, labels = list(dataset.train_dataset.take(1).as_numpy_iterator())[0]
                predictions = model.predict(imgs)
                tensorboard.write_predictions(imgs, predictions, labels, epoch, mode="Train")

                # Metrics for the Validation dataset
                imgs, labels = list(dataset.val_dataset.take(1).as_numpy_iterator())[0]
                predictions = model.predict(imgs)
                tensorboard.write_predictions(imgs, predictions, labels, epoch, mode="Validation")

            print(f"\nValidation loss: {val_loss}, Validation accuracy: {val_acc}",
                  f"-  Took {time.time() - validation_start_time:.5f}s", flush=True)
