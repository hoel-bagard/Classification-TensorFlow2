import tensorflow as tf


def CNN(input_shape, output_classes):
    # inputs = tf.keras.Input(shape=input_shape, name="InputLayer")
    inputs = tf.keras.Input(shape=(28, 28, 1,), name="InputLayer")
    # x = tf.keras.layers.InputLayer(input_shape=input_shape)(inputs)
    x = tf.keras.layers.Conv2D(16, 3, strides=2, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu", name="LastConv")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(output_classes, activation="softmax")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)
