import tensorflow as tf
from tensorflow.keras.layers import Conv2D

from src.layers.my_conv2D import MyConv2D
from src.layers.inverted_residual import InvertedResidual


def SmallMobileNet(input_shape, output_classes):
    inputs = tf.keras.Input(shape=input_shape, name="InputLayer")
    x = MyConv2D(filters=12, kernel_size=3, strides=2, padding="same")(inputs)
    x = InvertedResidual(filters=16, strides=2, expansion_factor=1)(x)
    x = InvertedResidual(filters=32, strides=2, expansion_factor=6)(x)
    # Adding this layers seems to make the network to deep/wide to learn
    # x = InvertedResidual(filters=64, strides=1, expansion_factor=6)(x)
    x = Conv2D(filters=output_classes, kernel_size=1, strides=1, padding="same")(x)
    x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(output_classes, activation="softmax")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


# class SmallMobileNet(tf.keras.Model):
#     def __init__(self, input_shape, output_classes):
#         super().__init__()
#         self.output_classes = output_classes

#         self.network = tf.keras.Sequential([
#             InputLayer(input_shape=input_shape),
#             MyConv2D(filters=12, kernel_size=3, strides=2, padding="same"),
#             InvertedResidual(filters=16, strides=2, expansion_factor=1),
#             InvertedResidual(filters=32, strides=2, expansion_factor=6),
#             InvertedResidual(filters=64, strides=1, expansion_factor=6),
#             Conv2D(filters=output_classes, kernel_size=1, strides=1, padding="same", activation="softmax"),

#             tf.keras.layers.Flatten(),
#             tf.keras.layers.Dropout(0.2),
#             tf.keras.layers.Dense(output_classes, activation="softmax")
#         ])

#     def call(self, inputs):
#         x = self.network(inputs)
#         # x = tf.squeeze(x)
#         return x
