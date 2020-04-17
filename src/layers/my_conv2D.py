import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    ReLU,
)
from config.model_config import ModelConfig


class MyConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding, trainable=True, name=None, **kwargs):
        super(MyConv2D, self).__init__(trainable=trainable, name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

    def build(self, input_shape):
        self.conv2D = Conv2D(filters=self.filters,
                             kernel_size=self.kernel_size,
                             strides=self.strides,
                             padding=self.padding,
                             kernel_regularizer=tf.keras.regularizers.l2(ModelConfig.REG_FACTOR),
                             activation=None)
        self.bn = BatchNormalization()
        self.relu6 = ReLU(max_value=6, negative_slope=0.1)

    def call(self, input_x):
        x = self.conv2D(input_x)
        x = self.bn(x)
        x = self.relu6(x)
        return x

    def get_config(self):
        cfg = super(MyConv2D, self).get_config()
        cfg.update({'filters': self.filters,
                    'kernel_size': self.kernel_size,
                    'strides': self.strides,
                    'padding': self.padding})
        return cfg
