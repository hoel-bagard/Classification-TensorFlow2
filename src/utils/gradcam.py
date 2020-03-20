import tensorflow as tf


def gradcam(layer_output_op: tf.Tensor, network_output_op: tf.Tensor, label_hot_op: tf.Tensor):
    gc_network_activation_op = tf.reduce_sum(network_output_op * label_hot_op, axis=-1)
    gc_gradient_op = tf.gradients(gc_network_activation_op, layer_output_op)[0]

    gc_weight_op = tf.reduce_mean(gc_gradient_op, axis=(1, 2))
    gc_layer_trans_op = tf.transpose(layer_output_op, [1, 2, 0, 3])
    return tf.nn.relu(tf.reduce_sum(
        tf.transpose(gc_weight_op * gc_layer_trans_op, [2, 0, 1, 3]),
        axis=-1))
