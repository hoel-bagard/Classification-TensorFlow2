import os

import cv2
import tensorflow as tf
import numpy as np
from PIL import Image

from src.utils.draw import draw_pred_mnist
from src.utils.gradcam import gradcam


def infer(input_files, checkpoint_path: str, output_dir: str, batch_size: int, use_gradcam: bool):
    imgs = []
    for i, file_path in enumerate(input_files):
        if i % batch_size == 0:
            imgs.append([])
        image = np.asarray(Image.open(file_path))
        imgs[-1].append(image)
    imgs = np.asarray(imgs)
    imgs = np.asarray(np.expand_dims(imgs, -1), dtype=np.float32)
    print(f"Running inference on test data : {imgs.shape}")

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    model = tf.keras.models.load_model(checkpoint_path)   # .expect_partial() ?
    print('Checkpoint restored')

    for i, imgs_batch in enumerate(imgs):
        print(f"Computing predictions for batch {i}")
        if use_gradcam:
            grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(name="Conv1").output, model.output])
            # conv_outputs, predictions = grad_model.predict_on_batch(imgs_batch)

            img = tf.keras.preprocessing.image.load_img(input_files[0], target_size=(28, 28))
            img = tf.keras.preprocessing.image.img_to_array(img)

            # img_temp = tf.expand_dims(imgs_batch[0], 0)
            # Get the score for target class
            with tf.GradientTape() as tape:
                # conv_output, predictions = grad_model(img_temp)
                conv_outputs, predictions = grad_model(np.array([img]))
                loss = predictions[:, 0]

                # Corentin's way of doing it
                # label_one_hot = tf.one_hot(tf.math.argmax(predictions, axis=-1), predictions.shape[-1])
                # gc_network_activation = tf.reduce_sum(predictions * label_one_hot, axis=-1)
            # gc_gradient = tape.gradient(gc_network_activation, conv_output)[0]
            # gc_weight = tf.reduce_mean(gc_gradient, axis=(1, 2))
            # gc_weight = tf.expand_dims(gc_weight, -1)
            # gc_layer_trans = tf.transpose(conv_output, [1, 2, 0, 3])
            # img = tf.nn.relu(tf.reduce_sum(
            #     tf.transpose(gc_weight * gc_layer_trans, [2, 0, 1, 3]),
            #     axis=-1))

            # Extract filters and gradients
            output = conv_outputs[0]
            grads = tape.gradient(loss, conv_outputs)[0]

            # Average gradients spatially
            weights = tf.reduce_mean(grads, axis=(0, 1))

            # Build a ponderated map of filters according to gradients importance
            cam = np.ones(output.shape[0:2], dtype=np.float32)

            for index, w in enumerate(weights):
                cam += w * output[:, :, index]

            # Heatmap visualization
            cam = cv2.resize(cam.numpy(), (224, 224))
            cam = np.maximum(cam, 0)
            heatmap = (cam - cam.min()) / (cam.max() - cam.min())

            cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)

            output_image = cv2.addWeighted(cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR), 0.5, cam, 1, 0)

            cv2.imwrite(os.path.join(output_dir, f"prediction_{i}.png"), output_image)

        else:
            predictions = model.predict_on_batch(imgs_batch)
            imgs_with_pred = draw_pred_mnist(imgs_batch, predictions)

            for j, img in enumerate(imgs_with_pred):
                img = img.astype(np.uint8)
                cv2.imwrite(os.path.join(output_dir, f"prediction_{i}_{j}.png"), img)

    print('Test done')
