import argparse
import glob
import os
from datetime import datetime
# import time    time / benchmark the thing

import numpy as np
import tensorflow as tf
from PIL import Image

from src.network import CNN


def load_data(data_path):
	imgs = []
	labels = []
	for i in range(10):
		for entry in glob.glob(os.path.join(data_path,str(i), "*.png"), recursive=True):
			print("Loading data {}".format(entry), end="\r")
			image = Image.open(entry)
			imgs.append(np.asarray(image))
			labels.append(i)
	try:
	    print('Data loaded' + str(' ' * (os.get_terminal_size()[0] - 11)))
	except:
		print("Data loaded" + ' '*40)
	return np.asarray(np.expand_dims(imgs, -1), dtype=np.float32), np.asarray(labels, dtype=np.uint8)

file_writer = tf.summary.create_file_writer(os.path.join("output", "logs", datetime.now().strftime("%Y%m%d-%H%M%S")))
def train(model, train_dataset, val_dataset, output_dir, steps_per_epoch, max_epoch=50):
    output_classes = 10
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-3)

    for epoch in range(max_epoch):
        print(f"\nEpoch {epoch}/{max_epoch}")
        epoch_loss = 0
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf.one_hot(y_batch_train, output_classes), logits))
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            epoch_loss += loss
            epoch_progress = int(30 * (step/steps_per_epoch))
            print(f"{step}/{steps_per_epoch} [" + epoch_progress*"=" + ">" + (30-epoch_progress)*"." + f"] - loss: {loss}" + 15*" ", end="\r", flush=True)

        # Also give how long it took
        print(f"\nEpoch loss: {epoch_loss/step}")
        
        acc_train = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=-1), tf.argmax(tf.one_hot(y_batch_train, output_classes), axis=-1)), tf.float32))
        with file_writer.as_default():
            tf.summary.scalar("Loss/Train",          loss,          step=epoch)
            tf.summary.scalar("Accuracy/Train",      acc_train,     step=epoch)
            tf.summary.image("Train", x_batch_train, max_outputs=4, step=epoch)
            file_writer.flush()

        # Validation every 10 epochs
        if epoch % 10 == 0 and epoch > 10:
            x_batch_val, y_batch_val = val_dataset.take(1)
            logits = model(x_batch_val, training=True)
            acc_train = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=-1), tf.argmax(tf.one_hot(x_batch_val, output_classes), axis=-1)), tf.float32))
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_batch_val, logits))
            with file_writer.as_default():
                tf.summary.scalar("Loss/Validation",          loss,          step=epoch)
                tf.summary.scalar("Accuracy/Validation",      acc_train,     step=epoch)
                file_writer.flush()


def main():
	parser = argparse.ArgumentParser("MNIST - TensorFlow 2")
	parser.add_argument("data_path", help="Path to the data folder")
	parser.add_argument("--output_dir", default="output", help="Output directory")
	parser.add_argument("--batch_size", default=512, help="Batch Size")
	args = parser.parse_args()

	train_data_path = os.path.join(args.data_path, "Train")
	val_data_path = os.path.join(args.data_path, "Test")

	x_train, y_train = load_data(train_data_path)
	x_val, y_val = load_data(val_data_path)
	print(f"Loaded {len(x_train)} train data and {len(x_val)} val data of shape {x_train.shape[1:]}")

	# Prepare the training dataset.
	train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
	train_dataset = train_dataset.shuffle(buffer_size=1024).batch(args.batch_size)

	# Prepare the validation dataset.
	val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
	val_dataset = train_dataset.shuffle(buffer_size=1024).batch(args.batch_size)

	model = CNN(x_train.shape[1:], 10)

	train(model, train_dataset, val_dataset, args.output_dir, len(x_train)//args.batch_size)


if __name__ == "__main__":
	main()