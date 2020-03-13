import time
import os

import tensorflow as tf

from src.dataset.mnist import MNISTDatasetCreator
from config.data_config import DataConfig
from config.model_config import ModelConfig
from src.utils.tensorboard import TensorBoard
from src.utils.trainer import Trainer


def train(model: tf.keras.Model, dataset: MNISTDatasetCreator, max_epoch=50):
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-3)
    loss_fn = tf.losses.SparseCategoricalCrossentropy(from_logits=False)
    trainer = Trainer(model, optimizer, loss_fn, dataset)

    if DataConfig.USE_TB:
        tensorboard = TensorBoard(model, dataset.input_shape, tb_dir=DataConfig.TB_DIR)
    best_loss = 1000

    for epoch in range(ModelConfig.MAX_EPOCHS):
        print(f"\nEpoch {epoch}/{max_epoch}")
        epoch_start_time = time.time()

        train_loss, train_acc = trainer.train_epoch()

        if DataConfig.USE_TB:
            tensorboard.write_metrics(train_loss, train_acc, epoch)
            tensorboard.write_lr(ModelConfig.LR, epoch)

        if train_loss < best_loss and DataConfig.USE_CHECKPOINT and epoch >= DataConfig.RECORD_DELAY:
            best_loss = train_loss
            model.save(os.path.join(DataConfig.CHECKPOINT_DIR, f'train_{epoch}'))

        print(f"\nEpoch loss: {train_loss}, Train accuracy: {train_acc}  -  Took {time.time() - epoch_start_time:.5f}s")

        # Validation
        if epoch % DataConfig.VAL_FREQ == 0 and epoch > DataConfig.RECORD_DELAY:
            val_loss, val_acc = trainer.val_epoch()
            print(f"Validation loss: {val_loss}, Validation accuracy: {val_acc}", flush=True)
            if DataConfig.USE_TB:
                tensorboard.write_metrics(train_loss, train_acc, epoch, mode="Validation")
                imgs, labels = list(dataset.val_dataset.shuffle(500).take(1).as_numpy_iterator())[0]
                predictions = model.predict(imgs)
                tensorboard.write_predictions(imgs, predictions, epoch, mode="Validation")
