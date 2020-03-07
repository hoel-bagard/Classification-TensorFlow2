import shutil
import os 
import sys
import stat

import tensorflow as tf
import numpy as np
import random

def train(train_data, train_label, output_dir, max_epoch, batch_size):

	train_data = np.expand_dims(train_data, -1)

	input_shape = train_data.shape[1:]
	output_classes = 10

	with tf.Session() as session:
		input_pl = tf.placeholder(dtype=tf.float32, shape=[None, *input_shape], name="input_pl")
		label_pl = tf.placeholder(dtype=tf.int32, shape=[None], name="label_pl")
		label_hot_op = tf.one_hot(label_pl, output_classes)

		with tf.variable_scope("Network"):
			network_output = tf.layers.conv2d(input_pl, 16, 3, strides=2, padding="same", activation=tf.contrib.keras.layers.LeakyReLU())
			network_output = tf.layers.conv2d(network_output, 32, 3, strides=2, padding="same", activation=tf.contrib.keras.layers.LeakyReLU())
			network_output = tf.layers.conv2d(network_output, 64, 3, strides=2, padding="same", activation=tf.contrib.keras.layers.LeakyReLU())
			network_output = tf.contrib.layers.flatten(network_output)
			network_output = tf.nn.dropout(network_output, rate=0.2)
			network_output = tf.contrib.layers.fully_connected(network_output, output_classes, activation_fn=None)

		tf_summaries = []
		tf_summaries.append(tf.summary.image("Image", input_pl, max_outputs=4))

		with tf.variable_scope("Train"):
			with tf.variable_scope("Loss"):
				loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(label_hot_op, network_output))
				tf_summaries.append(tf.summary.scalar("Loss", loss))
			with tf.variable_scope("Optimizer"):
				train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
			with tf.variable_scope("Accuracy"):
				accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(network_output, axis=-1), tf.argmax(label_hot_op, axis=-1)), tf.float32))
				tf_summaries.append(tf.summary.scalar("Accuracy", accuracy))

		summary_op = tf.summary.merge(tf_summaries)
		session.run(tf.global_variables_initializer())

		shutil.rmtree(output_dir, ignore_errors=True)
		os.makedirs(output_dir)
		tf_saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)
		tf_writter = tf.summary.FileWriter(os.path.join(output_dir, "tensorboard"), graph=session.graph, flush_secs=10)

		tf_saver.save(session, os.path.join(output_dir, "checkpoints"), global_step=0)

		global_step = 0
		data_length = train_data.shape[0]
		steps_per_epoch = data_length // batch_size
		data_indexes = list(range(data_length))
		for epoch in range(max_epoch):
			random.shuffle(data_indexes)
			for step in range(steps_per_epoch):
				image_batch = train_data[data_indexes[step * batch_size: (step+1) * batch_size]]
				label_batch = train_label[data_indexes[step * batch_size: (step+1) * batch_size]]

				session.run(train_op, feed_dict={input_pl: image_batch, label_pl: label_batch})
				global_step += 1

			summaries = session.run(summary_op, feed_dict={input_pl: image_batch, label_pl: label_batch})
			tf_writter.add_summary(summaries, global_step=global_step)
			print(global_step)
			sys.stdout.flush()


	print("So far so good")