import cv2
import numpy as np
import os
from glob import glob

def build_background(image_paths, noise_std=0):
    n = len(image_paths)
    bg_min, bg_max, bg_sum = None, None, None

    if isinstance(image_paths, str) and os.path.isdir(image_paths):
        image_paths = sorted(
            glob(os.path.join(image_paths, "*.jpg")) +
            glob(os.path.join(image_paths, "*.png")) +
            glob(os.path.join(image_paths, "*.jpeg"))
        )

    for i, p in enumerate(image_paths):
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB).astype(np.float32)
        if noise_std > 0:
            img += np.random.randn(*img.shape) * noise_std

        if i == 0:
            bg_min, bg_max, bg_sum = img.copy(), img.copy(), img.copy()
        else:
            bg_min = np.minimum(bg_min, img)
            bg_max = np.maximum(bg_max, img)
            bg_sum += img

    bg_mean = bg_sum / n
    return {"mean": bg_mean, "min": bg_min, "max": bg_max}

def save_background(model, save_path):
    np.savez(save_path, **model)

