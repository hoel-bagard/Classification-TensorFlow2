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
            grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(name="LastConv").output, model.output])
            conv_outputs, predictions = grad_model.predict_on_batch(imgs_batch)

            img_temp = tf.expand_dims(imgs_batch[0], 0)
            # Get the score for target class
            with tf.GradientTape() as tape:
                conv_output, predictions = grad_model(img_temp)
                loss = predictions[:, np.argmax(predictions[0])]
                grads = tape.gradient(loss, conv_output)[0]  #remove the [0]
                # pooled_grads = tf.keras.backend.mean(grads, axis=(0, 1, 2))

            # Average gradients spatially
            weights = tf.reduce_mean(grads, axis=(0, 1))

            # Build a ponderated map of filters according to gradients importance
            cam = np.ones(conv_output.shape[0:2], dtype=np.float32)

            for index, w in enumerate(weights):
                cam += w * conv_output[:, :, index]

            # Heatmap visualization
            cam = cv2.resize(cam.numpy(), (224, 224))
            cam = np.maximum(cam, 0)
            heatmap = (cam - cam.min()) / (cam.max() - cam.min())

            cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)

            output_image = cv2.addWeighted(cv2.cvtColor(imgs_batch[0].astype('uint8'), cv2.COLOR_RGB2BGR), 0.5, cam, 1, 0)

            cv2.imwrite(os.path.join(output_dir, f"prediction_{i}.png"), output_image)

        else:
            predictions = model.predict_on_batch(imgs_batch)
            imgs_with_pred = draw_pred_mnist(imgs_batch, predictions)

            for j, img in enumerate(imgs_with_pred):
                img = img.astype(np.uint8)
                cv2.imwrite(os.path.join(output_dir, f"prediction_{i}_{j}.png"), img)

    print('Test done')
