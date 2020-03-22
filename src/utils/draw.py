import cv2
import numpy as np


def draw_pred(imgs: np.ndarray, predictions: np.ndarray):
    new_imgs = []
    for img, preds in zip(imgs, predictions):
        img = cv2.resize(img, (480, 480), interpolation=cv2.INTER_AREA)
        preds = str([round(float(conf), 2) for conf in preds]) + f"  ==> {np.argmax(preds)}"
        new_imgs.append(cv2.putText(img, preds, (20, 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA))
    return np.asarray(new_imgs)
