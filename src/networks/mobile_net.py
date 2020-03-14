import tensorflow as tf
from tensorflow.keras.layers import (
    AveragePooling2D,
    Conv2D,
    InputLayer,
)

from src.layers.my_conv2D import MyConv2D
from src.layers.inverted_residual import InvertedResidual


class MobileNetV2(tf.keras.Model):
    # https://arxiv.org/pdf/1801.04381.pdf
    def __init__(self, input_shape, output_classes):
        super().__init__()
        self.output_classes = output_classes
        exp = [1, 6, 6, 6, 6, 6, 6]  # expansion_factor
        channels = [32, 16, 24, 32,  64, 96, 160, 320]
        repeats = [1, 2, 3, 4, 3, 3, 1]  # Number of times the layer is repeated
        strides = [1, 2, 2, 2, 1, 2, 1]

        self.network = tf.keras.Sequential([
            InputLayer(input_shape=input_shape),
            MyConv2D(filters=16, kernel_size=3, strides=2, padding="same"),
            *[InvertedResidual(filters=channels[i],
                               strides=strides[i],
                               expansion_factor=exp[i]) for i in range(len(exp)) for _ in range(repeats[i])],
            MyConv2D(filters=1280, kernel_size=3, strides=1, padding="same"),
            AveragePooling2D(1),
            Conv2D(filters=output_classes, kernel_size=1, strides=1, padding="same", activation=None),
        ])

    def call(self, inputs):
        x = self.network(inputs)
        x = tf.squeeze(x)
        return x
