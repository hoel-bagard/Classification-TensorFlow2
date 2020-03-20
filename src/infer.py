import os

import cv2
import tensorflow as tf
import numpy as np
from PIL import Image

from src.utils.draw import draw_pred_mnist


def infer(input_files, checkpoint_path, output_dir, batch_size):
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
        predictions = model.predict_on_batch(imgs_batch)
        imgs_with_pred = draw_pred_mnist(imgs_batch, predictions)

        for j, img in enumerate(imgs_with_pred):
            img = img.astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, f"prediction_{i}_{j}.png"), img)

    print('Test done')
