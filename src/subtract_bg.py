import cv2
import numpy as np

def load_background(load_path):
    data = np.load(load_path)
    return {k: data[k] for k in data.files}

def subtract_background(image_path, bg_model, threshold):
    """
    compaeres the input image to the background model and returns a binary mask
    """
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB).astype(np.float32)
    diff = np.linalg.norm(img - bg_model["mean"], axis=2)

    mask = (diff > threshold).astype(np.uint8) * 255
    return mask


