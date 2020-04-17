import os

import cv2
import tensorflow as tf
import numpy as np
from PIL import Image

from src.utils.draw import draw_pred
from src.utils.gradcam import gradcam
from config.model_config import ModelConfig


def infer(input_files, checkpoint_path: str, output_dir: str, batch_size: int, use_gradcam: bool):
    imgs = []
    for i, file_path in enumerate(input_files):
        if i % batch_size == 0:
            imgs.append([])
        image = Image.open(file_path)
        image = np.asarray(image.resize((ModelConfig.IMG_SIZE, ModelConfig.IMG_SIZE)))
        imgs[-1].append(image)
    imgs = np.asarray(imgs)
    if imgs.shape[-1] == 1:  # Black and white imgs (MNIST)
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
            grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(index=-5).output, model.output])
            # conv_outputs, predictions = grad_model.predict_on_batch(imgs_batch)

            # Get the score for target class
            with tf.GradientTape() as tape:
                # conv_output, predictions = grad_model(img_temp)
                conv_outputs, predictions = grad_model(imgs_batch)
                loss = predictions[:, 0]

                # Corentin's way of doing it
            #     label_one_hot = tf.one_hot(tf.math.argmax(predictions, axis=-1), predictions.shape[-1])
            #     gc_network_activation = tf.reduce_sum(predictions * label_one_hot, axis=-1)
            # gc_gradient = tape.gradient(gc_network_activation, conv_outputs)
            # gc_weight = tf.reduce_mean(gc_gradient, axis=(1, 2))
            # gc_layer_trans = tf.transpose(conv_outputs, [1, 2, 0, 3])
            # gradcam_imgs = tf.nn.relu(tf.reduce_sum(
            #     tf.transpose(gc_weight * gc_layer_trans, [2, 0, 1, 3]),
            #     axis=-1))

            # Other way of doing the grad-CAM
            gradcam_imgs = []
            # Extract filters and gradients
            output_batch = conv_outputs
            grads_batch = tape.gradient(loss, conv_outputs)
            for k in range(len(imgs_batch)):
                output = output_batch[k]
                grads = grads_batch[k]

                # Average gradients spatially
                weights = tf.reduce_mean(grads, axis=(0, 1))

                # Build a ponderated map of filters according to gradients importance
                cam = np.ones(output.shape[0:2], dtype=np.float32)

                for index, w in enumerate(weights):
                    cam += w * output[:, :, index]
                gradcam_imgs.append(cam)

            # Add heatmap and save image
            for j, image in enumerate(gradcam_imgs):
                # cv2.imwrite(os.path.join(output_dir, f"prediction_{i}_{j}.png"), np.float32(image))

                # # Heatmap visualization
                cam = cv2.resize(image.numpy(), (ModelConfig.IMG_SIZE, ModelConfig.IMG_SIZE))
                cam = np.maximum(cam, 0)
                heatmap = (cam - cam.min()) / (cam.max() - cam.min())

                cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
                input_image = cv2.cvtColor(imgs_batch[j].astype('uint8'), cv2.COLOR_RGB2BGR)
                output_image = cv2.addWeighted(input_image, 0.5, cam, 1, 0)
                cv2.imwrite(os.path.join(output_dir, f"prediction_{i}_{j}.png"), output_image)

        else:
            predictions = model.predict_on_batch(imgs_batch)
            imgs_with_pred = draw_pred(imgs_batch, predictions)

            for j, img in enumerate(imgs_with_pred):
                img = img.astype(np.uint8)
                cv2.imwrite(os.path.join(output_dir, f"prediction_{i}_{j}.png"), img)

    print('Test done')
