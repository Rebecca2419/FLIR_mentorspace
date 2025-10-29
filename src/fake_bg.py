import cv2
import numpy as np


def build_background(image_paths, noise_std=0):
    imgs = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB).astype(np.float32)
            for p in image_paths]

    if noise_std > 0:
        imgs = [img + np.random.randn(*img.shape) * noise_std for img in imgs]

    imgs = np.stack(imgs, axis=0)
    bg_mean = np.mean(imgs, axis=0)
    bg_min  = np.min(imgs, axis=0)
    bg_max  = np.max(imgs, axis=0)

    return {"mean": bg_mean, "min": bg_min, "max": bg_max}

def save_background(model, save_path):
    np.savez(save_path, **model)


