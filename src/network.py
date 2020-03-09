import tensorflow as tf


class CNN(tf.keras.Model):
    def __init__(self, input_shape, output_classes):
        super(CNN, self).__init__()
        self.network = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Conv2D(16, 3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(output_classes)
        ])

    def call(self, inputs):
        x = self.network(inputs)
        return x
