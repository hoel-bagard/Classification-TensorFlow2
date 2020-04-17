import tensorflow as tf

from src.layers.my_conv2D import MyConv2D


def CNN(input_shape, output_classes):
    # input_shape = 128
    inputs = tf.keras.Input(shape=input_shape, name="InputLayer")
    x = MyConv2D(16, 3, strides=3, padding="same", name="Conv1")(inputs)
    x = MyConv2D(24, 2, strides=2, padding="same", name="Conv2")(x)
    x = MyConv2D(32, 3, strides=3, padding="same", name="Conv3")(x)
    x = MyConv2D(16, 3, strides=3, padding="same", name="Conv4")(x)
    x = MyConv2D(12, 3, strides=3, padding="same", name="Conv5")(x)
    x = tf.keras.layers.Conv2D(output_classes, 1, 1, padding="same", activation="softmax", name="Conv6")(x)
    outputs = tf.keras.layers.Flatten()(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)
